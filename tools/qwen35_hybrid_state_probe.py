"""State normalization helpers for the Qwen3.5 compatibility gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import typing
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent


def _load_contract():
    module_name = "qwen35_hybrid_state_contract_for_probe"
    spec = importlib.util.spec_from_file_location(
        module_name,
        os.fspath(THIS_DIR / "qwen35_hybrid_state_contract.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract()

STATE_SNAPSHOT_FIELDS = (
    "snapshot_id",
    "request_id",
    "request_generation",
    "lifetime_epoch",
    "sequence_length",
    "component_count",
    "component_sha256",
)
MEMORY_SNAPSHOT_FIELDS = (
    "snapshot_id",
    "phase",
    "request_id",
    "request_generation",
    "cuda_allocated_bytes",
    "cuda_reserved_bytes",
    "logical_state_bytes",
    "unique_storage_bytes",
)
EXPORT_SCHEMA = "qwen35_hybrid_state_components"


class IncompleteRun(RuntimeError):
    def __init__(self, failure_kind, detail):
        super().__init__(f"{failure_kind}: {detail}")
        self.failure_kind = failure_kind
        self.detail = detail


_ORIGINAL_CUSTOM_OP_INFER_SCHEMA = None


def _resolve_custom_op_schema(function, mutates_args=()):
    original = _ORIGINAL_CUSTOM_OP_INFER_SCHEMA
    if original is None:
        raise RuntimeError("custom-op schema compatibility is not active")
    annotations = getattr(function, "__annotations__", None)
    if not annotations or not any(
        isinstance(value, str) for value in annotations.values()
    ):
        return original(function, mutates_args)
    resolved = typing.get_type_hints(
        function,
        globalns=function.__globals__,
    )
    function.__annotations__ = resolved
    try:
        return original(function, mutates_args)
    finally:
        function.__annotations__ = annotations


@contextmanager
def torch_custom_op_annotation_compatibility(
    *,
    infer_schema_owner=None,
):
    global _ORIGINAL_CUSTOM_OP_INFER_SCHEMA
    if infer_schema_owner is None:
        import torch._custom_op.impl as infer_schema_owner
    if _ORIGINAL_CUSTOM_OP_INFER_SCHEMA is not None:
        raise RuntimeError("custom-op schema compatibility is nested")
    original = infer_schema_owner.infer_schema
    _ORIGINAL_CUSTOM_OP_INFER_SCHEMA = original
    infer_schema_owner.infer_schema = _resolve_custom_op_schema
    try:
        yield
    finally:
        infer_schema_owner.infer_schema = original
        _ORIGINAL_CUSTOM_OP_INFER_SCHEMA = None


def _architecture_incomplete(detail):
    raise IncompleteRun("INCOMPLETE_ARCHITECTURE", detail)


def _model_layers(model):
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "layers", None),
        getattr(model, "layers", None),
    )
    for layers in candidates:
        if layers is not None:
            return tuple(layers)
    _architecture_incomplete("unable to locate decoder layers")


def _normalized_layer_type(value):
    text = str(value).lower().replace("-", "_")
    if "full" in text and "attention" in text:
        return "full_attention"
    if "linear" in text or "delta" in text or "mamba" in text:
        return "linear_attention"
    _architecture_incomplete(f"unsupported layer type: {value}")


def _layer_type(layer, config_layer_type):
    for candidate in (
        getattr(layer, "layer_type", None),
        getattr(getattr(layer, "self_attn", None), "layer_type", None),
        getattr(getattr(layer, "self_attn", None), "attention_type", None),
        config_layer_type,
    ):
        if candidate is not None:
            return _normalized_layer_type(candidate)
    _architecture_incomplete(
        f"layer {type(layer).__name__} exposes no explicit layer type"
    )


def inspect_model(*, model, config, tokenizer):
    layers = _model_layers(model)
    architecture_config = getattr(config, "text_config", config)
    tokenizer_vocab_size = int(tokenizer.vocab_size)
    model_vocab_size = int(
        getattr(architecture_config, "vocab_size", tokenizer_vocab_size)
    )
    if model_vocab_size < tokenizer_vocab_size:
        _architecture_incomplete(
            "model vocabulary is smaller than tokenizer vocabulary"
        )
    configured_types = getattr(architecture_config, "layer_types", None)
    if configured_types is not None and len(configured_types) != len(layers):
        _architecture_incomplete(
            "config layer_types length does not match loaded model"
        )
    layer_types = []
    for index, layer in enumerate(layers):
        configured = (
            configured_types[index] if configured_types is not None else None
        )
        layer_types.append(_layer_type(layer, configured))
    parameter_dtypes = {}
    for _, parameter in model.named_parameters():
        dtype = _dtype_name(parameter.dtype)
        parameter_dtypes[dtype] = (
            parameter_dtypes.get(dtype, 0) + int(parameter.numel())
        )
    result = {
        "config_class": type(config).__name__,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "model_vocab_size": model_vocab_size,
        "num_hidden_layers": int(
            getattr(architecture_config, "num_hidden_layers", len(layers))
        ),
        "linear_attention_layers": layer_types.count("linear_attention"),
        "full_attention_layers": layer_types.count("full_attention"),
        "full_attention_interval": int(
            getattr(architecture_config, "full_attention_interval", 0)
        ),
        "linear_num_key_heads": int(
            getattr(architecture_config, "linear_num_key_heads", -1)
        ),
        "linear_num_value_heads": int(
            getattr(architecture_config, "linear_num_value_heads", -1)
        ),
        "linear_key_head_dim": int(
            getattr(architecture_config, "linear_key_head_dim", -1)
        ),
        "linear_value_head_dim": int(
            getattr(architecture_config, "linear_value_head_dim", -1)
        ),
        "linear_conv_kernel_dim": int(
            getattr(architecture_config, "linear_conv_kernel_dim", -1)
        ),
        "mamba_ssm_dtype": _dtype_name(
            getattr(architecture_config, "mamba_ssm_dtype", "")
        ),
        "layer_schedule": {
            str(index): layer_type
            for index, layer_type in enumerate(layer_types)
        },
        "parameter_dtypes": dict(sorted(parameter_dtypes.items())),
    }
    require_canonical_architecture(result)
    return result


def _require_parameter_dtype(architecture, requested_dtype):
    parameter_dtypes = architecture.get("parameter_dtypes")
    if not isinstance(parameter_dtypes, Mapping) or not parameter_dtypes:
        raise IncompleteRun(
            "INCOMPLETE_MODEL_LOAD",
            "loaded model parameter dtype inventory is missing",
        )
    observed = {
        _dtype_name(dtype)
        for dtype, count in parameter_dtypes.items()
        if isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
    }
    expected = _dtype_name(requested_dtype)
    if observed != {expected}:
        raise IncompleteRun(
            "INCOMPLETE_MODEL_LOAD",
            "loaded model parameter dtypes do not match requested dtype: "
            f"requested {expected}, observed {sorted(observed)}",
        )


def load_official_reference(
    model_dir,
    *,
    requested_dtype="bfloat16",
    auto_config=None,
    auto_tokenizer=None,
    auto_model=None,
    custom_op_compatibility=torch_custom_op_annotation_compatibility,
):
    if auto_config is None or auto_tokenizer is None or auto_model is None:
        try:
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise IncompleteRun(
                "INCOMPLETE_RUNTIME",
                f"transformers import failed: {exc}",
            ) from exc
        auto_config = auto_config or AutoConfig
        auto_tokenizer = auto_tokenizer or AutoTokenizer
        auto_model = auto_model or AutoModelForCausalLM
    model_path = Path(model_dir)
    if not model_path.is_absolute():
        _reference_incomplete("model_dir must be an absolute path")
    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if requested_dtype not in dtype_by_name:
        _reference_incomplete(
            f"unsupported requested model dtype: {requested_dtype}"
        )
    try:
        config = auto_config.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        tokenizer = auto_tokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        with custom_op_compatibility():
            model = auto_model.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=False,
                dtype=dtype_by_name[requested_dtype],
            )
            model = model.to(
                device="cuda:0",
                dtype=dtype_by_name[requested_dtype],
            )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise IncompleteRun(
            "INCOMPLETE_MODEL_LOAD",
            f"official local model load failed: {type(exc).__name__}: {exc}",
        ) from exc
    architecture = inspect_model(
        model=model,
        config=config,
        tokenizer=tokenizer,
    )
    _require_parameter_dtype(architecture, requested_dtype)
    return {
        "config": config,
        "tokenizer": tokenizer,
        "model": model,
        "architecture": architecture,
        "requested_model_dtype": requested_dtype,
        "adapter": ReferenceStateAdapter(
            model=model,
            layer_schedule=architecture["layer_schedule"],
            vocab_size=architecture["tokenizer_vocab_size"],
            model_vocab_size=architecture["model_vocab_size"],
            device="cuda:0",
            requested_model_dtype=requested_dtype,
            parameter_dtypes=architecture["parameter_dtypes"],
        ),
    }


def require_canonical_architecture(architecture):
    expected_scalars = {
        "num_hidden_layers": contract.EXPECTED_NUM_HIDDEN_LAYERS,
        "linear_attention_layers": contract.EXPECTED_LINEAR_LAYERS,
        "full_attention_layers": contract.EXPECTED_FULL_ATTENTION_LAYERS,
        "full_attention_interval": contract.EXPECTED_FULL_ATTENTION_INTERVAL,
        "linear_num_key_heads": contract.EXPECTED_LINEAR_NUM_KEY_HEADS,
        "linear_num_value_heads": contract.EXPECTED_LINEAR_NUM_VALUE_HEADS,
        "linear_key_head_dim": contract.EXPECTED_LINEAR_KEY_HEAD_DIM,
        "linear_value_head_dim": contract.EXPECTED_LINEAR_VALUE_HEAD_DIM,
        "linear_conv_kernel_dim": contract.EXPECTED_LINEAR_CONV_KERNEL_DIM,
        "mamba_ssm_dtype": contract.EXPECTED_MAMBA_SSM_DTYPE,
    }
    mismatches = []
    for field, expected in expected_scalars.items():
        actual = architecture.get(field)
        if actual != expected:
            mismatches.append(f"{field}: expected {expected}, got {actual}")
    schedule = architecture.get("layer_schedule")
    if not isinstance(schedule, Mapping):
        mismatches.append("layer_schedule must be a mapping")
    else:
        if len(schedule) != contract.EXPECTED_NUM_HIDDEN_LAYERS:
            mismatches.append(
                "layer_schedule length: expected "
                f"{contract.EXPECTED_NUM_HIDDEN_LAYERS}, got {len(schedule)}"
            )
        observed = [schedule.get(str(index)) for index in range(
            contract.EXPECTED_NUM_HIDDEN_LAYERS
        )]
        if any(value not in {"linear_attention", "full_attention"}
               for value in observed):
            mismatches.append("layer_schedule contains missing/unknown entries")
        full_indices = [
            index for index, value in enumerate(observed)
            if value == "full_attention"
        ]
        expected_full_indices = list(
            range(
                contract.EXPECTED_FULL_ATTENTION_INTERVAL - 1,
                contract.EXPECTED_NUM_HIDDEN_LAYERS,
                contract.EXPECTED_FULL_ATTENTION_INTERVAL,
            )
        )
        if full_indices != expected_full_indices:
            mismatches.append(
                "full-attention schedule mismatch: "
                f"expected {expected_full_indices}, got {full_indices}"
            )
    if mismatches:
        _architecture_incomplete("; ".join(mismatches))
    return architecture


def _architecture_identity(architecture):
    return {
        key: value
        for key, value in architecture.items()
        if key != "parameter_dtypes"
    }


def capture_memory_snapshot(
    *,
    snapshot_id,
    phase,
    request_id,
    request_generation,
    components,
    cuda=None,
):
    cuda_api = cuda or torch.cuda
    if not cuda_api.is_available():
        raise IncompleteRun(
            "INCOMPLETE_RUNTIME",
            "CUDA is unavailable for allocator snapshot",
        )
    cuda_api.synchronize()
    return {
        "snapshot_id": str(snapshot_id),
        "phase": str(phase),
        "request_id": str(request_id),
        "request_generation": int(request_generation),
        "cuda_allocated_bytes": int(cuda_api.memory_allocated()),
        "cuda_reserved_bytes": int(cuda_api.memory_reserved()),
        "logical_state_bytes": sum(
            int(component["logical_bytes"])
            for component in components
        ),
        "unique_storage_bytes": contract.unique_storage_bytes(components),
    }


def _reference_incomplete(detail):
    raise IncompleteRun("INCOMPLETE_REFERENCE_SEMANTICS", detail)


def _encode_tensor_tree(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().cpu()
        return {
            "__kind__": "tensor",
            "dtype": _dtype_name(tensor.dtype),
            "shape": list(tensor.shape),
            "data": tensor.tolist(),
        }
    if isinstance(value, tuple):
        return {
            "__kind__": "tuple",
            "items": [_encode_tensor_tree(item) for item in value],
        }
    if isinstance(value, list):
        return {
            "__kind__": "list",
            "items": [_encode_tensor_tree(item) for item in value],
        }
    if isinstance(value, Mapping):
        return {
            "__kind__": "mapping",
            "items": [
                [str(key), _encode_tensor_tree(value[key])]
                for key in sorted(value, key=lambda item: str(item))
            ],
        }
    if value is None or isinstance(value, (str, bool, int, float)):
        return {"__kind__": "scalar", "value": value}
    _reference_incomplete(
        f"unsupported cache payload leaf: {type(value).__name__}"
    )


def _decode_tensor_tree(value):
    kind = value.get("__kind__")
    if kind == "tensor":
        dtype_name = value["dtype"]
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            _reference_incomplete(f"unsupported tensor dtype: {dtype_name}")
        tensor = torch.tensor(value["data"], dtype=dtype)
        return tensor.reshape(tuple(value["shape"]))
    if kind == "tuple":
        return tuple(_decode_tensor_tree(item) for item in value["items"])
    if kind == "list":
        return [_decode_tensor_tree(item) for item in value["items"]]
    if kind == "mapping":
        return {
            key: _decode_tensor_tree(item)
            for key, item in value["items"]
        }
    if kind == "scalar":
        return value["value"]
    _reference_incomplete(f"unsupported encoded cache kind: {kind}")


class ReferenceStateAdapter:
    def __init__(
        self,
        *,
        model,
        layer_schedule,
        vocab_size,
        model_vocab_size=None,
        device="cuda:0",
        requested_model_dtype=None,
        parameter_dtypes=None,
    ):
        self.model = model
        self.layer_schedule = {
            int(index): layer_type
            for index, layer_type in layer_schedule.items()
        }
        self.vocab_size = int(vocab_size)
        if self.vocab_size <= 1:
            _reference_incomplete("vocab_size must be greater than one")
        self.model_vocab_size = int(
            self.vocab_size
            if model_vocab_size is None
            else model_vocab_size
        )
        if self.model_vocab_size < self.vocab_size:
            _reference_incomplete(
                "model_vocab_size must cover tokenizer vocabulary"
            )
        self.device = torch.device(device)
        self._cache_type = None
        self.requested_model_dtype = (
            None
            if requested_model_dtype is None
            else _dtype_name(requested_model_dtype)
        )
        self.parameter_dtypes = (
            None
            if parameter_dtypes is None
            else dict(parameter_dtypes)
        )

    def _hybrid_layer_state(self, state):
        layers = getattr(state, "layers", None)
        if not isinstance(layers, Sequence):
            _reference_incomplete("hybrid cache has no layer sequence")
        if len(layers) != len(self.layer_schedule):
            _reference_incomplete("hybrid cache layer count mismatch")
        result = {}
        for index, layer in enumerate(layers):
            layer_type = self.layer_schedule.get(index)
            if layer_type == "linear_attention":
                conv_states = getattr(layer, "conv_states", None)
                recurrent_states = getattr(layer, "recurrent_states", None)
                if not isinstance(conv_states, torch.Tensor):
                    _reference_incomplete(
                        f"linear layer {index} has no convolution state"
                    )
                if not isinstance(recurrent_states, torch.Tensor):
                    _reference_incomplete(
                        f"linear layer {index} has no recurrent state"
                    )
                result[str(index)] = {
                    "linear_convolution_state": conv_states,
                    "linear_recurrent_state": recurrent_states,
                }
            elif layer_type == "full_attention":
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if not isinstance(keys, torch.Tensor):
                    _reference_incomplete(
                        f"attention layer {index} has no key state"
                    )
                if not isinstance(values, torch.Tensor):
                    _reference_incomplete(
                        f"attention layer {index} has no value state"
                    )
                result[str(index)] = {
                    "full_attention_key": keys,
                    "full_attention_value": values,
                }
            else:
                _reference_incomplete(
                    f"unsupported cache layer type at {index}: {layer_type}"
                )
        return {"layers": result}

    def _forward(self, input_ids, state, sequence_length):
        values = input_ids.to(device=self.device, dtype=torch.long)
        start = int(sequence_length)
        cache_position = torch.arange(
            start,
            start + values.shape[-1],
            device=self.device,
            dtype=torch.long,
        )
        try:
            output = self.model(
                input_ids=values,
                past_key_values=state,
                use_cache=True,
                return_dict=True,
                cache_position=cache_position,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            _reference_incomplete(
                f"official cache forward failed: {type(exc).__name__}: {exc}"
            )
        if not hasattr(output, "logits"):
            _reference_incomplete("model output has no logits")
        if not hasattr(output, "past_key_values"):
            _reference_incomplete("model output has no past_key_values")
        cache = output.past_key_values
        if cache is None:
            _reference_incomplete("model returned no persistent cache")
        if self._cache_type is None:
            self._cache_type = type(cache)
        elif type(cache) is not self._cache_type:
            _reference_incomplete("cache type changed during continuation")
        return _canonical_logits(output.logits), cache

    def prefill(self, input_ids, state):
        if not isinstance(input_ids, torch.Tensor):
            _reference_incomplete("prefill input_ids must be a tensor")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            _reference_incomplete(
                "prefill currently requires one rank-2 request tensor"
            )
        sequence_length = 0 if state is None else self._state_length(state)
        return self._forward(input_ids, state, sequence_length)

    def decode_one(self, token_id, state, sequence_length):
        if state is None:
            _reference_incomplete("decode_one requires non-empty state")
        observed = self._state_length(state)
        if observed is not None and observed != int(sequence_length):
            _reference_incomplete(
                "cache sequence length mismatch: "
                f"expected {sequence_length}, got {observed}"
            )
        return self._forward(
            torch.tensor([[int(token_id)]], dtype=torch.long),
            state,
            int(sequence_length),
        )

    def one_shot(self, token_ids):
        values = tuple(int(token_id) for token_id in token_ids)
        if not values:
            _reference_incomplete("one_shot requires at least one token")
        if any(
            token_id < 0 or token_id >= self.model_vocab_size
            for token_id in values
        ):
            _reference_incomplete("one_shot token is outside vocabulary")
        logits, _ = self._forward(
            torch.tensor([values], dtype=torch.long),
            None,
            0,
        )
        return logits

    def _state_length(self, state):
        for name in ("get_seq_length", "get_usable_length"):
            method = getattr(state, name, None)
            if callable(method):
                try:
                    return int(method())
                except TypeError:
                    continue
        tokens = getattr(state, "tokens", None)
        if isinstance(tokens, torch.Tensor):
            return int(tokens.numel())
        return None

    def export_state(
        self,
        state,
        request_id,
        request_generation,
        sequence_length,
    ):
        method = getattr(state, "to_legacy_cache", None)
        cache_type = type(state)
        constructor = getattr(cache_type, "from_legacy_cache", None)
        observed = self._state_length(state)
        if observed is not None and observed != int(sequence_length):
            _reference_incomplete(
                "export cache sequence length mismatch"
            )
        if callable(method) and callable(constructor):
            cache_codec = "legacy_cache"
            payload = _encode_tensor_tree(method())
        else:
            cache_codec = "hybrid_layers_v1"
            payload = _encode_tensor_tree(self._hybrid_layer_state(state))
        return {
            "schema": "qwen35_reference_cache",
            "schema_version": contract.SCHEMA_VERSION,
            "cache_codec": cache_codec,
            "cache_class_module": cache_type.__module__,
            "cache_class_name": cache_type.__qualname__,
            "request_id": str(request_id),
            "request_generation": int(request_generation),
            "sequence_length": int(sequence_length),
            "layer_schedule": {
                str(index): layer_type
                for index, layer_type in sorted(self.layer_schedule.items())
            },
            "payload": payload,
            "payload_sha256": contract.canonical_json_sha256(payload),
        }

    def import_state(self, exported):
        if exported.get("schema") != "qwen35_reference_cache":
            _reference_incomplete("unsupported exported cache schema")
        if exported.get("schema_version") != contract.SCHEMA_VERSION:
            _reference_incomplete("unsupported exported cache version")
        cache_codec = exported.get("cache_codec")
        if cache_codec not in {"legacy_cache", "hybrid_layers_v1"}:
            _reference_incomplete("unsupported cache codec")
        expected_schedule = {
            str(index): layer_type
            for index, layer_type in sorted(self.layer_schedule.items())
        }
        if exported.get("layer_schedule") != expected_schedule:
            _reference_incomplete("exported layer schedule mismatch")
        payload = exported.get("payload")
        if exported.get("payload_sha256") != (
            contract.canonical_json_sha256(payload)
        ):
            _reference_incomplete("exported cache payload hash mismatch")
        cache_type = self._cache_type
        if cache_type is None:
            _reference_incomplete("cache type is not bound")
        if (
            exported.get("cache_class_module") != cache_type.__module__
            or exported.get("cache_class_name") != cache_type.__qualname__
        ):
            _reference_incomplete("exported cache class mismatch")
        constructor = getattr(cache_type, "from_legacy_cache", None)
        if cache_codec == "legacy_cache":
            if not callable(constructor):
                _reference_incomplete(
                    "cache type cannot import legacy payload"
                )
            try:
                return constructor(_decode_tensor_tree(payload))
            except (TypeError, ValueError, RuntimeError) as exc:
                _reference_incomplete(
                    f"cache import failed: {type(exc).__name__}: {exc}"
                )
        decoded = _decode_tensor_tree(payload)
        layer_payloads = decoded.get("layers")
        if not isinstance(layer_payloads, Mapping):
            _reference_incomplete("hybrid cache payload has no layers")
        try:
            restored = cache_type(config=self.model.config)
        except (TypeError, ValueError, RuntimeError) as exc:
            _reference_incomplete(
                f"hybrid cache construction failed: "
                f"{type(exc).__name__}: {exc}"
            )
        if len(getattr(restored, "layers", ())) != len(self.layer_schedule):
            _reference_incomplete("restored hybrid cache layer count mismatch")
        for index, layer_type in sorted(self.layer_schedule.items()):
            layer_payload = layer_payloads.get(str(index))
            if not isinstance(layer_payload, Mapping):
                _reference_incomplete(
                    f"hybrid cache payload missing layer {index}"
                )
            try:
                if layer_type == "linear_attention":
                    conv_states = layer_payload[
                        "linear_convolution_state"
                    ].to(self.device)
                    recurrent_states = layer_payload[
                        "linear_recurrent_state"
                    ].to(self.device)
                    restored.update_conv_state(conv_states, index)
                    restored.update_recurrent_state(
                        recurrent_states,
                        index,
                    )
                elif layer_type == "full_attention":
                    keys = layer_payload["full_attention_key"].to(self.device)
                    values = layer_payload[
                        "full_attention_value"
                    ].to(self.device)
                    restored.update(keys, values, index)
                else:
                    _reference_incomplete(
                        f"unsupported cache layer type at "
                        f"{index}: {layer_type}"
                    )
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                _reference_incomplete(
                    f"hybrid cache layer {index} import failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        return restored

    def state_sha256(self, state):
        exported = self.export_state(
            state,
            request_id="state-hash",
            request_generation=0,
            sequence_length=self._state_length(state) or 0,
        )
        return exported["payload_sha256"]

    def state_for_normalization(self, state):
        method = getattr(state, "to_legacy_cache", None)
        if callable(method):
            payload = method()
            if not isinstance(payload, Sequence):
                _reference_incomplete(
                    "legacy cache payload is not layer-indexed"
                )
            return {
                "layers": {
                    str(index): layer_state
                    for index, layer_state in enumerate(payload)
                }
            }
        return self._hybrid_layer_state(state)


def _canonical_logits(logits):
    if not isinstance(logits, torch.Tensor):
        _reference_incomplete("reference logits are not a torch.Tensor")
    value = logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if value.ndim == 0:
        _reference_incomplete("reference logits must have a vocabulary axis")
    if value.ndim > 1:
        value = value.reshape(-1, value.shape[-1])[-1]
    if value.numel() == 0:
        _reference_incomplete("reference logits are empty")
    if not bool(torch.isfinite(value).all()):
        _reference_incomplete("reference logits contain non-finite values")
    return value


def _logit_sha256(logits):
    value = _canonical_logits(logits)
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def _logit_differences(actual, expected):
    actual_value = _canonical_logits(actual)
    expected_value = _canonical_logits(expected)
    if actual_value.shape != expected_value.shape:
        _reference_incomplete(
            "reference logit shapes differ: "
            f"{tuple(actual_value.shape)} != {tuple(expected_value.shape)}"
        )
    absolute = (actual_value - expected_value).abs()
    denominator = expected_value.abs().clamp_min(
        contract.MIN_LOGIT_TOLERANCE
    )
    relative = absolute / denominator
    return {
        "max_abs_diff": float(absolute.max().item()),
        "mean_abs_diff": float(absolute.mean().item()),
        "max_rel_diff": float(relative.max().item()),
        "mean_rel_diff": float(relative.mean().item()),
    }


def _ranked_topk(logits):
    value = _canonical_logits(logits)
    if value.numel() < contract.DECISION_TOPK:
        _reference_incomplete(
            "reference vocabulary is smaller than DECISION_TOPK"
        )
    top_values, top_indices = torch.topk(
        value,
        k=contract.DECISION_TOPK,
    )
    token_ids = [int(token_id) for token_id in top_indices.tolist()]
    values = [float(logit) for logit in top_values.tolist()]
    try:
        contract.validate_ranked_topk(
            token_ids,
            values,
            require_positive_margin=False,
        )
    except ValueError as exc:
        raise IncompleteRun(
            "INCOMPLETE_REFERENCE_SEMANTICS",
            f"invalid winner evidence: {exc}",
        ) from exc
    return token_ids, values


def _comparison_metrics(actual, oracle):
    actual_value = _canonical_logits(actual)
    oracle_value = _canonical_logits(oracle)
    if actual_value.shape != oracle_value.shape:
        _reference_incomplete(
            "reference logit shapes differ: "
            f"{tuple(actual_value.shape)} != {tuple(oracle_value.shape)}"
        )
    absolute = (actual_value - oracle_value).abs()
    threshold = (
        contract.FP32_ATOL
        + contract.FP32_RTOL * oracle_value.abs()
    )
    scaled = absolute / threshold.clamp_min(
        torch.finfo(torch.float32).tiny
    )
    quantiles = torch.quantile(
        absolute,
        torch.tensor(
            [0.5, 0.95, 0.99, 0.999],
            dtype=torch.float32,
        ),
    )
    cosine = torch.nn.functional.cosine_similarity(
        actual_value.reshape(1, -1),
        oracle_value.reshape(1, -1),
    )
    cosine = cosine.clamp(min=-1.0, max=1.0)
    return {
        "abs_diff_percentiles": {
            name: float(value)
            for name, value in zip(
                contract.ABS_DIFF_PERCENTILE_FIELDS,
                quantiles.tolist(),
            )
        },
        "cosine_similarity": float(cosine.item()),
        "allclose_violation_count": int(
            (absolute > threshold).sum().item()
        ),
        "max_allclose_scaled_error": float(scaled.max().item()),
    }


def _logit_record(
    *,
    logits,
    oracle_logits,
    request_id,
    request_generation,
    step_index,
    sequence_length,
    comparison_policy="bf16_decision_preserving",
):
    value = _canonical_logits(logits)
    oracle_value = _canonical_logits(oracle_logits)
    actual_ids, actual_logits = _ranked_topk(value)
    oracle_ids, oracle_logits_values = _ranked_topk(oracle_value)
    actual_winner = contract.winner_margin(
        actual_ids,
        actual_logits,
        require_positive_margin=False,
    )
    oracle_winner = contract.winner_margin(
        oracle_ids,
        oracle_logits_values,
        require_positive_margin=False,
    )
    differences = _logit_differences(value, oracle_logits)
    comparison = _comparison_metrics(value, oracle_value)
    intersection_size = len(set(actual_ids).intersection(oracle_ids))
    return {
        "request_id": request_id,
        "request_generation": int(request_generation),
        "step_index": int(step_index),
        "full_logit_sha256": _logit_sha256(value),
        "topk_token_ids": actual_ids,
        "topk_logits": actual_logits,
        **differences,
        "sequence_length": int(sequence_length),
        "position_metadata": {
            "last_position": int(sequence_length) - 1,
            "actual_greedy_token_id": _greedy_token(value),
            "oracle_greedy_token_id": _greedy_token(oracle_value),
            "actual_full_logit_sha256": _logit_sha256(value),
            "oracle_full_logit_sha256": _logit_sha256(oracle_value),
            "comparison_policy": comparison_policy,
            "actual_logit_dtype": _dtype_name(logits.dtype),
            "oracle_logit_dtype": _dtype_name(oracle_logits.dtype),
        },
        "actual_topk_token_ids": actual_ids,
        "actual_topk_logits": actual_logits,
        "oracle_topk_token_ids": oracle_ids,
        "oracle_topk_logits": oracle_logits_values,
        "topk_intersection_size": intersection_size,
        "oracle_topk_recall": (
            intersection_size / contract.DECISION_TOPK
        ),
        "actual_winner_token_id": actual_winner["winner_token_id"],
        "oracle_winner_token_id": oracle_winner["winner_token_id"],
        "actual_runner_up_token_id": (
            actual_winner["runner_up_token_id"]
        ),
        "oracle_runner_up_token_id": (
            oracle_winner["runner_up_token_id"]
        ),
        "actual_winner_logit": actual_winner["winner_logit"],
        "oracle_winner_logit": oracle_winner["winner_logit"],
        "actual_runner_up_logit": actual_winner["runner_up_logit"],
        "oracle_runner_up_logit": oracle_winner["runner_up_logit"],
        "actual_winner_margin": actual_winner["winner_margin"],
        "oracle_winner_margin": oracle_winner["winner_margin"],
        "winner_logit_abs_diff": abs(
            actual_winner["winner_logit"]
            - oracle_winner["winner_logit"]
        ),
        "runner_up_logit_abs_diff": abs(
            actual_winner["runner_up_logit"]
            - oracle_winner["runner_up_logit"]
        ),
        "winner_margin_abs_diff": abs(
            actual_winner["winner_margin"]
            - oracle_winner["winner_margin"]
        ),
        **comparison,
    }


def _greedy_token(logits):
    return int(torch.argmax(_canonical_logits(logits)).item())


def _one_shot_logits(adapter, token_ids):
    method = getattr(adapter, "one_shot", None)
    if method is None:
        _reference_incomplete("adapter has no explicit one_shot method")
    return method(tuple(int(token_id) for token_id in token_ids))


def _state_snapshot_id(
    adapter,
    state,
    *,
    request_id,
    request_generation,
    sequence_length,
    snapshot_index,
):
    state_hash = getattr(adapter, "state_sha256", None)
    if state_hash is None:
        _reference_incomplete(
            "adapter has no explicit state_sha256 method"
        )
    return (
        f"{request_id}:g{request_generation}:s{snapshot_index}:"
        f"l{sequence_length}:{state_hash(state)}"
    )


def run_one_shot_oracle(
    adapter,
    *,
    token_ids,
    decode_steps,
    request_id="request-0",
    request_generation=0,
    comparison_policy="bf16_decision_preserving",
):
    tokens = [int(token_id) for token_id in token_ids]
    decoded = []
    records = []
    for step_index in range(int(decode_steps)):
        logits = _one_shot_logits(adapter, tokens)
        records.append(_logit_record(
            logits=logits,
            oracle_logits=logits,
            request_id=request_id,
            request_generation=request_generation,
            step_index=step_index,
            sequence_length=len(tokens),
            comparison_policy=comparison_policy,
        ))
        token_id = _greedy_token(logits)
        decoded.append(token_id)
        tokens.append(token_id)
    return {
        "request_id": request_id,
        "request_generation": int(request_generation),
        "decoded_token_ids": decoded,
        "logit_records": records,
        "state_snapshot_ids": [],
    }


def run_cached_decode(
    adapter,
    *,
    token_ids,
    decode_steps,
    request_id="request-0",
    request_generation=0,
    state_observer=None,
    comparison_policy="bf16_decision_preserving",
):
    prompt = tuple(int(token_id) for token_id in token_ids)
    input_ids = torch.tensor([prompt], dtype=torch.long)
    logits, state = adapter.prefill(input_ids, None)
    sequence_length = len(prompt)
    if state_observer is not None:
        state_observer(
            state=state,
            request_id=request_id,
            request_generation=request_generation,
            sequence_length=sequence_length,
            phase="after_prefill",
        )
    snapshots = [_state_snapshot_id(
        adapter,
        state,
        request_id=request_id,
        request_generation=request_generation,
        sequence_length=sequence_length,
        snapshot_index=0,
    )]
    tokens = list(prompt)
    decoded = []
    records = []
    for step_index in range(int(decode_steps)):
        oracle_logits = _one_shot_logits(adapter, tokens)
        records.append(_logit_record(
            logits=logits,
            oracle_logits=oracle_logits,
            request_id=request_id,
            request_generation=request_generation,
            step_index=step_index,
            sequence_length=sequence_length,
            comparison_policy=comparison_policy,
        ))
        token_id = _greedy_token(logits)
        decoded.append(token_id)
        tokens.append(token_id)
        logits, state = adapter.decode_one(
            token_id,
            state,
            sequence_length,
        )
        sequence_length += 1
        if state_observer is not None:
            state_observer(
                state=state,
                request_id=request_id,
                request_generation=request_generation,
                sequence_length=sequence_length,
                phase="after_decode",
            )
        snapshots.append(_state_snapshot_id(
            adapter,
            state,
            request_id=request_id,
            request_generation=request_generation,
            sequence_length=sequence_length,
            snapshot_index=len(snapshots),
        ))
    return {
        "request_id": request_id,
        "request_generation": int(request_generation),
        "decoded_token_ids": decoded,
        "logit_records": records,
        "state_snapshot_ids": snapshots,
    }


def run_chunked_prefill_decode(
    adapter,
    *,
    token_ids,
    chunk_schedule,
    decode_steps,
    request_id="request-0",
    request_generation=0,
    state_observer=None,
    comparison_policy="bf16_decision_preserving",
):
    prompt = tuple(int(token_id) for token_id in token_ids)
    schedule = tuple(int(chunk) for chunk in chunk_schedule)
    if not schedule or any(chunk <= 0 for chunk in schedule):
        raise ValueError("chunk_schedule must contain positive chunks")
    if sum(schedule) != len(prompt):
        raise ValueError("chunk_schedule must cover token_ids exactly")
    cursor = 0
    sequence_length = 0
    state = None
    logits = None
    snapshots = []
    for chunk in schedule:
        values = prompt[cursor:cursor + chunk]
        logits, state = adapter.prefill(
            torch.tensor([values], dtype=torch.long),
            state,
        )
        cursor += chunk
        sequence_length += chunk
        if state_observer is not None:
            state_observer(
                state=state,
                request_id=request_id,
                request_generation=request_generation,
                sequence_length=sequence_length,
                phase="after_prefill_chunk",
            )
        snapshots.append(_state_snapshot_id(
            adapter,
            state,
            request_id=request_id,
            request_generation=request_generation,
            sequence_length=sequence_length,
            snapshot_index=len(snapshots),
        ))
    tokens = list(prompt)
    decoded = []
    records = []
    for step_index in range(int(decode_steps)):
        oracle_logits = _one_shot_logits(adapter, tokens)
        records.append(_logit_record(
            logits=logits,
            oracle_logits=oracle_logits,
            request_id=request_id,
            request_generation=request_generation,
            step_index=step_index,
            sequence_length=sequence_length,
            comparison_policy=comparison_policy,
        ))
        token_id = _greedy_token(logits)
        decoded.append(token_id)
        tokens.append(token_id)
        logits, state = adapter.decode_one(
            token_id,
            state,
            sequence_length,
        )
        sequence_length += 1
        if state_observer is not None:
            state_observer(
                state=state,
                request_id=request_id,
                request_generation=request_generation,
                sequence_length=sequence_length,
                phase="after_decode",
            )
        snapshots.append(_state_snapshot_id(
            adapter,
            state,
            request_id=request_id,
            request_generation=request_generation,
            sequence_length=sequence_length,
            snapshot_index=len(snapshots),
        ))
    return {
        "request_id": request_id,
        "request_generation": int(request_generation),
        "decoded_token_ids": decoded,
        "logit_records": records,
        "state_snapshot_ids": snapshots,
    }


def run_export_import_continuation(
    adapter,
    *,
    token_ids,
    request_id="request-0",
    request_generation=0,
):
    prompt = tuple(int(token_id) for token_id in token_ids)
    logits, state = adapter.prefill(
        torch.tensor([prompt], dtype=torch.long),
        None,
    )
    exported = adapter.export_state(
        state,
        request_id,
        request_generation,
        len(prompt),
    )
    restored = adapter.import_state(exported)
    token_id = _greedy_token(logits)
    original_logits, _ = adapter.decode_one(
        token_id,
        state,
        len(prompt),
    )
    restored_logits, _ = adapter.decode_one(
        token_id,
        restored,
        len(prompt),
    )
    differences = _logit_differences(original_logits, restored_logits)
    original_next = _greedy_token(original_logits)
    restored_next = _greedy_token(restored_logits)
    return {
        "decoded_token_ids_equal": original_next == restored_next,
        "full_logit_sha256_equal": (
            _logit_sha256(original_logits)
            == _logit_sha256(restored_logits)
        ),
        "original_decoded_token_id": original_next,
        "restored_decoded_token_id": restored_next,
        **differences,
    }


def run_interleaved_requests(
    adapter,
    *,
    request_token_ids,
    replacement_token_ids,
    decode_steps=2,
    state_observer=None,
    perform_slot_reuse=True,
    comparison_policy="bf16_decision_preserving",
):
    request_ids = tuple(sorted(request_token_ids))
    if len(request_ids) != 3:
        raise ValueError("interleaving requires exactly three requests")
    if "slot-0" not in request_token_ids:
        raise ValueError("interleaving requires reusable slot-0")
    states = {}
    sequences = {}
    pending_logits = {}
    generations = {request_id: 0 for request_id in request_ids}
    slot_generations = {
        request_id: [0] for request_id in request_ids
    }
    for request_id in request_ids:
        tokens = tuple(
            int(token_id) for token_id in request_token_ids[request_id]
        )
        logits, state = adapter.prefill(
            torch.tensor([tokens], dtype=torch.long),
            None,
        )
        states[request_id] = state
        sequences[request_id] = list(tokens)
        pending_logits[request_id] = logits
        if state_observer is not None:
            state_observer(
                state=state,
                request_id=request_id,
                request_generation=0,
                sequence_length=len(tokens),
                phase="after_prefill",
            )

    inactive_changes = []
    serial_mismatches = []
    stale_reads = []
    released = []
    retired_hashes = set()
    decoded = []
    records = []
    decode_indices = {
        (request_id, generations[request_id]): 0
        for request_id in request_ids
    }
    active_schedule = (
        "slot-0",
        "slot-1",
        "slot-2",
        "slot-1",
        "slot-2",
    )

    def decode_active(request_id):
        inactive_before = {
            inactive: adapter.state_sha256(states[inactive])
            for inactive in sorted(states)
            if inactive != request_id
        }
        logits = pending_logits[request_id]
        oracle_logits = _one_shot_logits(
            adapter, sequences[request_id]
        )
        generation = generations[request_id]
        decode_key = (request_id, generation)
        step_index = decode_indices.setdefault(decode_key, 0)
        records.append(_logit_record(
            logits=logits,
            oracle_logits=oracle_logits,
            request_id=request_id,
            request_generation=generation,
            step_index=step_index,
            sequence_length=len(sequences[request_id]),
            comparison_policy=comparison_policy,
        ))
        token_id = _greedy_token(logits)
        decoded.append(token_id)
        decode_indices[decode_key] = step_index + 1
        if token_id != _greedy_token(oracle_logits):
            serial_mismatches.append([
                request_id,
                generations[request_id],
                len(sequences[request_id]),
            ])
        state_hash = adapter.state_sha256(states[request_id])
        if state_hash in retired_hashes:
            stale_reads.append([
                request_id,
                generations[request_id],
                state_hash,
            ])
        pending_logits[request_id], states[request_id] = adapter.decode_one(
            token_id,
            states[request_id],
            len(sequences[request_id]),
        )
        sequences[request_id].append(token_id)
        if state_observer is not None:
            state_observer(
                state=states[request_id],
                request_id=request_id,
                request_generation=generations[request_id],
                sequence_length=len(sequences[request_id]),
                phase="after_decode",
            )
        inactive_after = {
            inactive: adapter.state_sha256(states[inactive])
            for inactive in sorted(states)
            if inactive != request_id
        }
        for inactive in sorted(inactive_before):
            if inactive_before[inactive] != inactive_after[inactive]:
                inactive_changes.append([
                    request_id,
                    inactive,
                    inactive_before[inactive],
                    inactive_after[inactive],
                ])

    if int(decode_steps) == 2:
        for request_id in active_schedule:
            decode_active(request_id)
    else:
        for _ in range(int(decode_steps)):
            for request_id in request_ids:
                decode_active(request_id)

    if not perform_slot_reuse:
        return {
            "inactive_request_hash_changes": inactive_changes,
            "serial_oracle_mismatches": serial_mismatches,
            "slot_generations": slot_generations,
            "released_generations": released,
            "stale_state_reads": stale_reads,
            "decoded_token_ids": decoded,
            "logit_records": records,
        }

    retired_hashes.add(adapter.state_sha256(states["slot-0"]))
    released.append(["slot-0", generations["slot-0"]])
    del states["slot-0"]
    del sequences["slot-0"]
    del pending_logits["slot-0"]
    gc.collect()

    generations["slot-0"] += 1
    slot_generations["slot-0"].append(generations["slot-0"])
    replacement = tuple(int(token_id) for token_id in replacement_token_ids)
    replacement_logits, replacement_state = adapter.prefill(
        torch.tensor([replacement], dtype=torch.long),
        None,
    )
    replacement_hash = adapter.state_sha256(replacement_state)
    if replacement_hash in retired_hashes:
        stale_reads.append([
            "slot-0",
            generations["slot-0"],
            replacement_hash,
        ])
    states["slot-0"] = replacement_state
    sequences["slot-0"] = list(replacement)
    pending_logits["slot-0"] = replacement_logits
    if state_observer is not None:
        state_observer(
            state=replacement_state,
            request_id="slot-0",
            request_generation=generations["slot-0"],
            sequence_length=len(replacement),
            phase="after_slot_reuse",
        )
    decode_active("slot-0")

    return {
        "inactive_request_hash_changes": inactive_changes,
        "serial_oracle_mismatches": serial_mismatches,
        "slot_generations": slot_generations,
        "released_generations": released,
        "stale_state_reads": stale_reads,
        "decoded_token_ids": decoded,
        "logit_records": records,
    }


def _join_path(prefix, component):
    if not prefix:
        return component
    if component.startswith("["):
        return f"{prefix}{component}"
    return f"{prefix}.{component}"


def walk_tensor_leaves(state, *, adapter_registry=None):
    """Yield ``(path, tensor)`` leaves using only frozen container rules."""
    registry = adapter_registry or {}

    def walk(value, path):
        if isinstance(value, torch.Tensor):
            yield path, value
            return
        adapter = registry.get(type(value))
        if adapter is not None:
            yield from walk(adapter(value), path)
            return
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                yield from walk(
                    getattr(value, field.name),
                    _join_path(path, field.name),
                )
            return
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            for name in value._fields:
                yield from walk(
                    getattr(value, name),
                    _join_path(path, name),
                )
            return
        if isinstance(value, Mapping):
            for key in sorted(value, key=lambda item: str(item)):
                yield from walk(
                    value[key],
                    _join_path(path, str(key)),
                )
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, item in enumerate(value):
                yield from walk(item, _join_path(path, f"[{index}]"))
            return
        if value is None or isinstance(
            value, (str, bytes, bytearray, bool, int, float)
        ):
            return
        raise ValueError(
            f"unsupported state object {type(value).__name__}; "
            "register an explicit adapter"
        )

    yield from walk(state, "")


def _path_tokens(tensor_path):
    return (
        tensor_path.lower()
        .replace("[", ".")
        .replace("]", "")
        .replace("-", "_")
        .split(".")
    )


def classify_state_role(
    tensor_path,
    *,
    declared_layer_type,
    component_name=None,
):
    text = "_".join(
        token
        for token in _path_tokens(component_name or tensor_path)
        if token
    )
    layer_type = str(declared_layer_type).lower()
    if any(name in text for name in ("position", "sequence", "cache_position")):
        return "position_or_sequence_metadata"
    if "conv" in text:
        return "linear_convolution_state"
    if any(name in text for name in ("recurrent", "ssm", "state_cache")):
        return "linear_recurrent_state"
    if any(name in text for name in ("value_cache", "value_state", "values")):
        return "full_attention_value"
    if any(name in text for name in ("key_cache", "key_state", "keys")):
        return "full_attention_key"
    if "full" in layer_type or "attention" in layer_type:
        if text.endswith("_key") or text == "key":
            return "full_attention_key"
        if text.endswith("_value") or text == "value":
            return "full_attention_value"
    return "other_persistent_state"


def _dtype_name(dtype):
    name = str(dtype)
    return name.removeprefix("torch.")


def _dtype_profile(
    *,
    requested_model_dtype,
    architecture,
    state_components,
    logit_dtype,
):
    parameter_dtypes = architecture.get("parameter_dtypes")
    if not isinstance(parameter_dtypes, Mapping) or not parameter_dtypes:
        _reference_incomplete("parameter dtype inventory is missing")
    normalized_counts = {}
    for dtype, count in parameter_dtypes.items():
        name = _dtype_name(dtype)
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
        ):
            _reference_incomplete("parameter dtype counts must be positive")
        normalized_counts[name] = normalized_counts.get(name, 0) + count
    dominant = max(
        sorted(normalized_counts),
        key=lambda name: normalized_counts[name],
    )
    recurrent = sorted({
        row["dtype"]
        for row in state_components
        if row["state_role"] in {
            "linear_recurrent_state",
            "linear_convolution_state",
        }
    })
    kv = sorted({
        row["dtype"]
        for row in state_components
        if row["state_role"] in {
            "full_attention_key",
            "full_attention_value",
        }
    })
    if not recurrent or not kv:
        _reference_incomplete(
            "dtype profile requires recurrent and KV state dtypes"
        )
    return {
        "requested_model_dtype": _dtype_name(requested_model_dtype),
        "dominant_parameter_dtype": dominant,
        "logit_dtype_before_comparison": _dtype_name(logit_dtype),
        "comparison_accumulator_dtype": "float32",
        "recurrent_state_dtypes": recurrent,
        "kv_state_dtypes": kv,
    }


def _tensor_content_sha256(tensor):
    value = tensor.detach().contiguous().cpu()
    try:
        content = value.numpy().tobytes(order="C")
    except (TypeError, RuntimeError):
        content = value.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(content).hexdigest()


def _layer_index(tensor_path, layer_schedule):
    tokens = _path_tokens(tensor_path)
    if "layers" in tokens:
        position = tokens.index("layers")
        if position + 1 < len(tokens):
            try:
                return int(tokens[position + 1])
            except ValueError:
                pass
    if len(layer_schedule) == 1:
        return next(iter(layer_schedule))
    return -1


def normalize_state_components(
    *,
    state,
    request_id,
    request_generation,
    sequence_length,
    lifetime_epoch,
    layer_schedule,
    adapter_registry=None,
    component_names=None,
):
    rows = []
    names = component_names or {}
    for tensor_path, tensor in walk_tensor_leaves(
        state,
        adapter_registry=adapter_registry,
    ):
        layer_index = _layer_index(tensor_path, layer_schedule)
        declared_layer_type = layer_schedule.get(layer_index, "unknown")
        storage = tensor.untyped_storage()
        device = str(tensor.device)
        storage_data_ptr = int(storage.data_ptr())
        storage_nbytes = int(storage.nbytes())
        storage_identity = contract.canonical_json_sha256({
            "device": device,
            "storage_data_ptr": storage_data_ptr,
            "storage_nbytes": storage_nbytes,
        })
        dtype = _dtype_name(tensor.dtype)
        component = contract.StateComponent(
            request_id=str(request_id),
            request_generation=int(request_generation),
            layer_index=int(layer_index),
            declared_layer_type=str(declared_layer_type),
            state_role=classify_state_role(
                tensor_path,
                declared_layer_type=declared_layer_type,
                component_name=names.get(tensor_path),
            ),
            tensor_path=tensor_path,
            shape=tuple(int(value) for value in tensor.shape),
            stride=tuple(int(value) for value in tensor.stride()),
            dtype=dtype,
            device=device,
            requires_grad=bool(tensor.requires_grad),
            logical_numel=int(tensor.numel()),
            logical_bytes=int(tensor.numel() * tensor.element_size()),
            storage_data_ptr=storage_data_ptr,
            storage_offset=int(tensor.storage_offset()),
            storage_nbytes=storage_nbytes,
            storage_identity=storage_identity,
            lifetime_epoch=int(lifetime_epoch),
            sequence_length=int(sequence_length),
            update_kind="created",
            content_sha256=_tensor_content_sha256(tensor),
        )
        rows.append(asdict(component))
    return sorted(rows, key=_component_sort_key)


def _component_key(component):
    return (
        component["request_id"],
        int(component["request_generation"]),
        int(component["layer_index"]),
        component["state_role"],
        component["tensor_path"],
    )


def _component_sort_key(component):
    return _component_key(component)


def _validate_generation_ownership(components):
    owners = {}
    for component in components:
        identity = (
            component["request_id"],
            int(component["layer_index"]),
            component["state_role"],
            component["tensor_path"],
        )
        generation = int(component["request_generation"])
        previous = owners.setdefault(identity, generation)
        if previous != generation:
            raise ValueError(
                "component maps to more than one request generation"
            )


def _transition_kind(previous, current):
    if current["storage_identity"] != previous["storage_identity"]:
        return "replaced"
    if (
        current["shape"] != previous["shape"]
        or current["storage_offset"] != previous["storage_offset"]
        or current["storage_nbytes"] != previous["storage_nbytes"]
        or current["logical_bytes"] != previous["logical_bytes"]
    ):
        return "grown"
    if current["content_sha256"] != previous["content_sha256"]:
        return "mutated_in_place"
    return "unchanged"


def compare_state_snapshots(previous, current):
    _validate_generation_ownership([*previous, *current])
    previous_by_key = {_component_key(row): row for row in previous}
    current_by_key = {_component_key(row): row for row in current}
    if len(previous_by_key) != len(previous) or len(current_by_key) != len(current):
        raise ValueError("duplicate component key")
    transitions = {}
    for key in sorted(set(previous_by_key) | set(current_by_key)):
        before = previous_by_key.get(key)
        after = current_by_key.get(key)
        if before is None:
            kind = "created"
        elif after is None:
            kind = "released"
        else:
            kind = _transition_kind(before, after)
        transitions[key] = kind
    return transitions


def export_normalized_state(components):
    ordered = sorted((dict(row) for row in components), key=_component_sort_key)
    _validate_component_rows(ordered)
    return {
        "schema": EXPORT_SCHEMA,
        "schema_version": contract.SCHEMA_VERSION,
        "components": ordered,
        "components_sha256": contract.canonical_json_sha256(ordered),
    }


def import_normalized_state(payload):
    if payload.get("schema") != EXPORT_SCHEMA:
        raise ValueError("unsupported schema")
    if payload.get("schema_version") != contract.SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    components = [dict(row) for row in payload.get("components", ())]
    _validate_component_rows(components)
    expected = payload.get("components_sha256")
    actual = contract.canonical_json_sha256(components)
    if expected != actual:
        raise ValueError("components_sha256 mismatch")
    return sorted(components, key=_component_sort_key)


def _validate_component_rows(components):
    expected_fields = set(contract.StateComponent.__dataclass_fields__)
    seen = set()
    for component in components:
        if set(component) != expected_fields:
            raise ValueError("state component fields do not match contract")
        key = _component_key(component)
        if key in seen:
            raise ValueError("duplicate component key")
        seen.add(key)
    _validate_generation_ownership(components)


def _write_atomic(path, content):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()


def write_json_atomic(path, payload):
    content = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    _write_atomic(path, f"{content}\n")


def write_jsonl_atomic(path, rows, *, required_fields):
    expected = set(required_fields)
    lines = []
    for row in rows:
        if set(row) != expected:
            raise ValueError("JSONL row fields do not match required_fields")
        lines.append(json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ))
    _write_atomic(path, "".join(f"{line}\n" for line in lines))


def emit_raw_probe_artifacts(
    *,
    run_dir,
    architecture,
    case_rows,
    state_snapshots,
    state_components,
    memory_snapshots,
    parameter_bytes,
    max_memory_allocated,
    max_memory_reserved,
    dtype_profiles=None,
):
    destination = Path(run_dir)
    components = list(state_components)
    logical_state_bytes = sum(
        int(component["logical_bytes"]) for component in components
    )
    unique_state_bytes = contract.unique_storage_bytes(components)
    allocator_observation = max(
        0,
        int(max_memory_allocated) - int(unique_state_bytes),
    )
    summary = {
        "schema_version": contract.SCHEMA_VERSION,
        "architecture": architecture,
        "case_row_count": len(case_rows),
        "state_snapshot_count": len(state_snapshots),
        "state_component_count": len(components),
        "memory_snapshot_count": len(memory_snapshots),
        "parameter_bytes": int(parameter_bytes),
        "state_logical_bytes": logical_state_bytes,
        "state_unique_storage_bytes": unique_state_bytes,
        "max_memory_allocated": int(max_memory_allocated),
        "max_memory_reserved": int(max_memory_reserved),
        "non_state_peak_allocator_observation_bytes": allocator_observation,
        "dtype_profiles": dict(dtype_profiles or {}),
        "claim_boundary": (
            "Allocator observations are not exact state bytes and this "
            "worker summary is not an authoritative classification."
        ),
    }
    write_jsonl_atomic(
        destination / "case_rows.jsonl",
        case_rows,
        required_fields=contract.CASE_ROW_FIELDS,
    )
    write_jsonl_atomic(
        destination / "state_snapshots.jsonl",
        state_snapshots,
        required_fields=STATE_SNAPSHOT_FIELDS,
    )
    write_jsonl_atomic(
        destination / "state_components.jsonl",
        components,
        required_fields=tuple(contract.StateComponent.__dataclass_fields__),
    )
    write_jsonl_atomic(
        destination / "memory_snapshots.jsonl",
        memory_snapshots,
        required_fields=MEMORY_SNAPSHOT_FIELDS,
    )
    write_json_atomic(destination / "summary.json", summary)
    return summary


def contract_file_sha256():
    return hashlib.sha256(
        (THIS_DIR / "qwen35_hybrid_state_contract.py").read_bytes()
    ).hexdigest()


def _require_contract_sha256(expected):
    actual = contract_file_sha256()
    if str(expected) != actual:
        raise IncompleteRun(
            "INCOMPLETE_CONTRACT_MISMATCH",
            f"expected contract SHA256 {expected}, got {actual}",
        )
    return actual


def _case_token_ids(adapter, prompt_length, seed):
    vocab_size = int(getattr(adapter, "vocab_size", 0))
    if vocab_size > 256:
        return contract.deterministic_token_ids(
            length=int(prompt_length),
            vocab_size=vocab_size,
            seed=int(seed),
            forbidden_ids=set(),
        )
    if vocab_size <= 1:
        _reference_incomplete(
            "adapter has no authoritative vocabulary size"
        )
    return tuple(
        1 + ((int(seed) + index * 104729) % (vocab_size - 1))
        for index in range(int(prompt_length))
    )


def _row_request_identity(case):
    if case.phase in {
        "interleaved_multi_request",
        "completion_release_slot_reuse",
    }:
        request_ids = ["slot-0", "slot-1", "slot-2"]
        generations = [0, 0, 0]
        if case.phase == "completion_release_slot_reuse":
            request_ids.append("slot-0")
            generations.append(1)
        return request_ids, generations
    return ["request-0"], [0]


class _StateEvidenceCollector:
    def __init__(self, *, adapter, layer_schedule, cuda):
        self.adapter = adapter
        self.layer_schedule = {
            int(index): layer_type
            for index, layer_type in layer_schedule.items()
        }
        self.cuda = cuda
        self.state_snapshots = []
        self.state_components = []
        self.memory_snapshots = []
        self._lifetime_epoch = 0
        self._previous_components = {}

    def _memory(self, *, snapshot_id, phase, request_id,
                request_generation, components):
        row = capture_memory_snapshot(
            snapshot_id=f"memory:{snapshot_id}",
            phase=phase,
            request_id=request_id,
            request_generation=request_generation,
            components=components,
            cuda=self.cuda,
        )
        self.memory_snapshots.append(row)
        return row["snapshot_id"]

    def capture_empty(
        self,
        *,
        case_id,
        request_id,
        request_generation,
        phase,
    ):
        lifetime_epoch = self._lifetime_epoch
        self._lifetime_epoch += 1
        snapshot_id = (
            f"{case_id}:{request_id}:g{request_generation}:"
            f"e{lifetime_epoch}:{phase}"
        )
        self.state_snapshots.append({
            "snapshot_id": snapshot_id,
            "request_id": request_id,
            "request_generation": int(request_generation),
            "lifetime_epoch": lifetime_epoch,
            "sequence_length": 0,
            "component_count": 0,
            "component_sha256": contract.canonical_json_sha256([]),
        })
        memory_id = self._memory(
            snapshot_id=snapshot_id,
            phase=phase,
            request_id=request_id,
            request_generation=request_generation,
            components=[],
        )
        return snapshot_id, memory_id

    def capture_allocator(
        self,
        *,
        case_id,
        request_id,
        request_generation,
        phase,
    ):
        snapshot_id = (
            f"{case_id}:{request_id}:g{request_generation}:"
            f"allocator:{phase}"
        )
        return self._memory(
            snapshot_id=snapshot_id,
            phase=phase,
            request_id=request_id,
            request_generation=request_generation,
            components=[],
        )

    def capture_release(
        self,
        *,
        case_id,
        request_id,
        request_generation,
        sequence_length,
        phase,
    ):
        previous_key = (request_id, int(request_generation))
        previous = self._previous_components.pop(previous_key, None)
        if not previous:
            raise IncompleteRun(
                "INCOMPLETE_REFERENCE_SEMANTICS",
                "release has no prior state components",
            )
        lifetime_epoch = self._lifetime_epoch
        self._lifetime_epoch += 1
        components = [
            dict(
                component,
                lifetime_epoch=lifetime_epoch,
                sequence_length=int(sequence_length),
                update_kind="released",
            )
            for component in previous
        ]
        snapshot_id = (
            f"{case_id}:{request_id}:g{request_generation}:"
            f"e{lifetime_epoch}:{phase}"
        )
        self.state_components.extend(components)
        self.state_snapshots.append({
            "snapshot_id": snapshot_id,
            "request_id": request_id,
            "request_generation": int(request_generation),
            "lifetime_epoch": lifetime_epoch,
            "sequence_length": int(sequence_length),
            "component_count": len(components),
            "component_sha256": contract.canonical_json_sha256(components),
        })
        memory_id = self._memory(
            snapshot_id=snapshot_id,
            phase=phase,
            request_id=request_id,
            request_generation=request_generation,
            components=[],
        )
        return snapshot_id, memory_id

    def capture(
        self,
        *,
        case_id,
        state,
        request_id,
        request_generation,
        sequence_length,
        phase,
    ):
        lifetime_epoch = self._lifetime_epoch
        self._lifetime_epoch += 1
        normalizer = getattr(
            self.adapter,
            "state_for_normalization",
            None,
        )
        normalized_state = normalizer(state) if callable(normalizer) else state
        components = normalize_state_components(
            state=normalized_state,
            request_id=request_id,
            request_generation=request_generation,
            sequence_length=sequence_length,
            lifetime_epoch=lifetime_epoch,
            layer_schedule=self.layer_schedule,
        )
        previous_key = (request_id, int(request_generation))
        previous = self._previous_components.get(previous_key, [])
        if previous:
            transitions = compare_state_snapshots(previous, components)
            components = [
                dict(
                    component,
                    update_kind=transitions.get(
                        _component_key(component),
                        "created",
                    ),
                )
                for component in components
            ]
        self._previous_components[previous_key] = components
        snapshot_id = (
            f"{case_id}:{request_id}:g{request_generation}:"
            f"e{lifetime_epoch}:{phase}"
        )
        self.state_components.extend(components)
        self.state_snapshots.append({
            "snapshot_id": snapshot_id,
            "request_id": request_id,
            "request_generation": int(request_generation),
            "lifetime_epoch": lifetime_epoch,
            "sequence_length": int(sequence_length),
            "component_count": len(components),
            "component_sha256": contract.canonical_json_sha256(components),
        })
        memory_id = self._memory(
            snapshot_id=snapshot_id,
            phase=phase,
            request_id=request_id,
            request_generation=request_generation,
            components=components,
        )
        return snapshot_id, memory_id


def _run_slot_reuse_case(case, adapter, observer):
    request_ids = ("slot-0", "slot-1", "slot-2")
    generations = {request_id: 0 for request_id in request_ids}
    sequences = {
        request_id: list(_case_token_ids(
            adapter,
            prompt_length,
            seed=prompt_length + index,
        ))
        for index, (request_id, prompt_length) in enumerate(zip(
            request_ids,
            contract.MULTI_REQUEST_LENGTHS,
        ))
    }
    states = {}
    pending_logits = {}
    for request_id in request_ids:
        observer.capture_empty(
            case_id=case.case_id,
            request_id=request_id,
            request_generation=0,
            phase="before_prefill",
        )
        pending_logits[request_id], states[request_id] = adapter.prefill(
            torch.tensor([sequences[request_id]], dtype=torch.long),
            None,
        )
        observer.capture(
            case_id=case.case_id,
            state=states[request_id],
            request_id=request_id,
            request_generation=0,
            sequence_length=len(sequences[request_id]),
            phase="after_prefill",
        )
    decoded = []
    records = []

    def decode_active(request_id, step_index):
        inactive_before = {
            inactive_id: adapter.state_sha256(states[inactive_id])
            for inactive_id in sorted(states)
            if inactive_id != request_id
        }
        logits = pending_logits[request_id]
        tokens = sequences[request_id]
        oracle_logits = _one_shot_logits(adapter, tokens)
        records.append(_logit_record(
            logits=logits,
            oracle_logits=oracle_logits,
            request_id=request_id,
            request_generation=generations[request_id],
            step_index=step_index,
            sequence_length=len(tokens),
            comparison_policy=case.comparison_policy,
        ))
        token_id = _greedy_token(logits)
        decoded.append(token_id)
        pending_logits[request_id], states[request_id] = adapter.decode_one(
            token_id,
            states[request_id],
            len(tokens),
        )
        tokens.append(token_id)
        observer.capture(
            case_id=case.case_id,
            state=states[request_id],
            request_id=request_id,
            request_generation=generations[request_id],
            sequence_length=len(tokens),
            phase=f"after_decode_step_{step_index}",
        )
        inactive_after = {
            inactive_id: adapter.state_sha256(states[inactive_id])
            for inactive_id in sorted(states)
            if inactive_id != request_id
        }
        if inactive_before != inactive_after:
            raise IncompleteRun(
                "INCOMPLETE_REFERENCE_SEMANTICS",
                f"decoding {request_id} mutated inactive request state",
            )

    for step_index in range(2):
        decode_active("slot-0", step_index)
    for request_id in ("slot-1", "slot-2"):
        for step_index in range(case.decode_steps):
            decode_active(request_id, step_index)

    retired_hash = adapter.state_sha256(states["slot-0"])
    retired_length = len(sequences["slot-0"])
    del states["slot-0"]
    del sequences["slot-0"]
    del pending_logits["slot-0"]
    gc.collect()
    observer.capture_release(
        case_id=case.case_id,
        request_id="slot-0",
        request_generation=0,
        sequence_length=retired_length,
        phase="after_request_release",
    )
    generations["slot-0"] = 1
    replacement = _case_token_ids(
        adapter,
        contract.SLOT_REUSE_PROMPT_LENGTH,
        seed=contract.SLOT_REUSE_PROMPT_LENGTH,
    )
    pending_logits["slot-0"], states["slot-0"] = adapter.prefill(
        torch.tensor([replacement], dtype=torch.long),
        None,
    )
    sequences["slot-0"] = list(replacement)
    if adapter.state_sha256(states["slot-0"]) == retired_hash:
        raise IncompleteRun(
            "INCOMPLETE_REFERENCE_SEMANTICS",
            "slot reuse retained the retired state hash",
        )
    observer.capture(
        case_id=case.case_id,
        state=states["slot-0"],
        request_id="slot-0",
        request_generation=1,
        sequence_length=len(replacement),
        phase="after_slot_reuse",
    )
    for step_index in range(case.decode_steps):
        decode_active("slot-0", step_index)
    return decoded, records


def _run_reference_case(case, adapter, observer):
    request_ids, generations = _row_request_identity(case)
    decoded = []
    records = []
    state_start = len(observer.state_snapshots)
    memory_start = len(observer.memory_snapshots)
    decode_observation_index = 0

    def observe(**kwargs):
        nonlocal decode_observation_index
        if kwargs.get("phase") == "after_decode":
            kwargs["phase"] = (
                f"after_decode_step_{decode_observation_index}"
            )
            decode_observation_index += 1
        observer.capture(case_id=case.case_id, **kwargs)

    if case.execution_mode in {"preflight", "inspect_model", "post_run_audit"}:
        pass
    elif case.execution_mode in {
        "cached_repeatability",
        "one_shot_vs_cached",
        "cached_vs_one_shot",
        "state_memory_ledger",
    }:
        tokens = _case_token_ids(
            adapter,
            case.prompt_length,
            seed=case.prompt_length,
        )
        observer.capture_empty(
            case_id=case.case_id,
            request_id="request-0",
            request_generation=0,
            phase="before_prefill",
        )
        result = run_cached_decode(
            adapter,
            token_ids=tokens,
            decode_steps=case.decode_steps,
            state_observer=observe,
            comparison_policy=case.comparison_policy,
        )
        decoded = result["decoded_token_ids"]
        records = result["logit_records"]
    elif case.execution_mode == "one_shot_vs_chunked":
        tokens = _case_token_ids(
            adapter,
            case.prompt_length,
            seed=case.prompt_length,
        )
        observer.capture_empty(
            case_id=case.case_id,
            request_id="request-0",
            request_generation=0,
            phase="before_prefill",
        )
        result = run_chunked_prefill_decode(
            adapter,
            token_ids=tokens,
            chunk_schedule=case.chunk_schedule,
            decode_steps=case.decode_steps,
            state_observer=observe,
            comparison_policy=case.comparison_policy,
        )
        decoded = result["decoded_token_ids"]
        records = result["logit_records"]
    elif case.execution_mode == "state_export_import":
        tokens = _case_token_ids(
            adapter,
            case.prompt_length,
            seed=case.prompt_length,
        )
        observer.capture_empty(
            case_id=case.case_id,
            request_id="request-0",
            request_generation=0,
            phase="before_prefill",
        )
        cached = run_cached_decode(
            adapter,
            token_ids=tokens,
            decode_steps=case.decode_steps,
            state_observer=observe,
            comparison_policy=case.comparison_policy,
        )
        continuation = run_export_import_continuation(
            adapter,
            token_ids=tokens,
        )
        if not (
            continuation["decoded_token_ids_equal"]
            and continuation["full_logit_sha256_equal"]
        ):
            raise IncompleteRun(
                "INCOMPLETE_REFERENCE_SEMANTICS",
                f"{case.case_id} export/import continuation diverged",
            )
        decoded = cached["decoded_token_ids"]
        records = cached["logit_records"]
    elif case.execution_mode == "interleaved_multi_request":
        request_token_ids = {
            f"slot-{index}": _case_token_ids(
                adapter,
                prompt_length,
                seed=prompt_length + index,
            )
            for index, prompt_length in enumerate(
                contract.MULTI_REQUEST_LENGTHS
            )
        }
        for request_id in sorted(request_token_ids):
            observer.capture_empty(
                case_id=case.case_id,
                request_id=request_id,
                request_generation=0,
                phase="before_prefill",
            )
        result = run_interleaved_requests(
            adapter,
            request_token_ids=request_token_ids,
            replacement_token_ids=_case_token_ids(
                adapter,
                contract.SLOT_REUSE_PROMPT_LENGTH,
                seed=contract.SLOT_REUSE_PROMPT_LENGTH,
            ),
            decode_steps=case.decode_steps,
            state_observer=observe,
            perform_slot_reuse=False,
            comparison_policy=case.comparison_policy,
        )
        if (
            result["inactive_request_hash_changes"]
            or result["serial_oracle_mismatches"]
            or result["stale_state_reads"]
        ):
            raise IncompleteRun(
                "INCOMPLETE_REFERENCE_SEMANTICS",
                f"{case.case_id} request isolation evidence failed",
            )
        decoded = result["decoded_token_ids"]
        records = result["logit_records"]
    elif case.execution_mode == "completion_release_slot_reuse":
        decoded, records = _run_slot_reuse_case(
            case,
            adapter,
            observer,
        )
    else:
        raise IncompleteRun(
            "INCOMPLETE_REFERENCE_SEMANTICS",
            f"unsupported execution mode: {case.execution_mode}",
        )
    state_rows = observer.state_snapshots[state_start:]
    memory_rows = observer.memory_snapshots[memory_start:]
    snapshot_ids = [row["snapshot_id"] for row in state_rows]
    memory_ids = [row["snapshot_id"] for row in memory_rows]
    if len(snapshot_ids) != case.expected_state_snapshots:
        raise IncompleteRun(
            "INCOMPLETE_REFERENCE_SEMANTICS",
            f"{case.case_id} emitted {len(snapshot_ids)} state snapshots; "
            f"expected {case.expected_state_snapshots}",
        )
    row = {
        "row_id": f"row:{case.case_id}",
        "case_id": case.case_id,
        "phase": case.phase,
        "execution_mode": case.execution_mode,
        "prompt_length": case.prompt_length,
        "chunk_schedule": list(case.chunk_schedule),
        "request_count": case.request_count,
        "decode_steps": case.decode_steps,
        "repeat_index": case.repeat_index,
        "request_ids": request_ids,
        "request_generations": generations,
        "decoded_token_ids": decoded,
        "logit_records": records,
        "state_snapshot_ids": snapshot_ids,
        "memory_snapshot_ids": memory_ids,
        "complete": True,
        "failure_kind": None,
        "failure_detail": None,
        "execution_dtype": case.execution_dtype,
        "comparison_policy": case.comparison_policy,
    }
    return row


def run_reference_case_matrix(
    *,
    adapter_factory,
    fp32_adapter_factory=None,
    architecture,
    run_dir,
    contract_sha256,
    parameter_bytes,
    cuda_module=None,
):
    _require_contract_sha256(contract_sha256)
    require_canonical_architecture(architecture)
    cuda_api = cuda_module or torch.cuda
    cases = contract.build_case_matrix()
    case_rows_by_id = {}
    max_memory_allocated = 0
    max_memory_reserved = 0
    observer = None
    dtype_profiles = {}

    def run_cases(adapter, selected):
        nonlocal observer
        nonlocal max_memory_allocated
        nonlocal max_memory_reserved
        if observer is None:
            observer = _StateEvidenceCollector(
                adapter=adapter,
                layer_schedule=architecture["layer_schedule"],
                cuda=cuda_api,
            )
        else:
            observer.adapter = adapter
        component_start = len(observer.state_components)
        selected_rows = []
        for case in selected:
            if cuda_api.is_available():
                cuda_api.reset_peak_memory_stats()
            row = _run_reference_case(
                case,
                adapter,
                observer,
            )
            case_rows_by_id[case.case_id] = row
            selected_rows.append(row)
            if cuda_api.is_available():
                max_memory_allocated = max(
                    max_memory_allocated,
                    int(cuda_api.max_memory_allocated()),
                )
                max_memory_reserved = max(
                    max_memory_reserved,
                    int(cuda_api.max_memory_reserved()),
                )
        requested_dtype = getattr(
            adapter,
            "requested_model_dtype",
            None,
        )
        parameter_dtypes = getattr(adapter, "parameter_dtypes", None)
        records = [
            record
            for row in selected_rows
            for record in row["logit_records"]
        ]
        if requested_dtype is not None and parameter_dtypes and records:
            dtype_profiles[requested_dtype] = _dtype_profile(
                requested_model_dtype=requested_dtype,
                architecture={"parameter_dtypes": parameter_dtypes},
                state_components=observer.state_components[
                    component_start:
                ],
                logit_dtype=records[0]["position_metadata"][
                    "actual_logit_dtype"
                ],
            )

    adapter = adapter_factory()
    run_cases(
        adapter,
        [
            case
            for case in cases
            if case.execution_dtype != "float32"
        ],
    )
    if observer is not None:
        observer.adapter = None
    del adapter
    gc.collect()
    if cuda_api.is_available() and hasattr(cuda_api, "empty_cache"):
        cuda_api.empty_cache()
    fp32_factory = fp32_adapter_factory or adapter_factory
    fp32_adapter = fp32_factory()
    run_cases(
        fp32_adapter,
        [
            case
            for case in cases
            if case.execution_dtype == "float32"
        ],
    )
    if observer is not None:
        observer.adapter = None
    del fp32_adapter
    gc.collect()
    if cuda_api.is_available() and hasattr(cuda_api, "empty_cache"):
        cuda_api.empty_cache()
    case_rows = [
        case_rows_by_id[case.case_id]
        for case in cases
    ]
    if observer is None:
        raise IncompleteRun(
            "INCOMPLETE_REFERENCE_SEMANTICS",
            "reference matrix emitted no cases",
        )
    write_json_atomic(Path(run_dir) / "architecture.json", architecture)
    summary = emit_raw_probe_artifacts(
        run_dir=run_dir,
        architecture=architecture,
        case_rows=case_rows,
        state_snapshots=observer.state_snapshots,
        state_components=observer.state_components,
        memory_snapshots=observer.memory_snapshots,
        parameter_bytes=parameter_bytes,
        max_memory_allocated=max_memory_allocated,
        max_memory_reserved=max_memory_reserved,
        dtype_profiles=dtype_profiles,
    )
    return {
        "case_rows": case_rows,
        "state_snapshots": observer.state_snapshots,
        "state_components": observer.state_components,
        "memory_snapshots": observer.memory_snapshots,
        "summary": summary,
        "dtype_profiles": dtype_profiles,
    }


def _loaded_reference_parts(loaded):
    if isinstance(loaded, Mapping):
        return (
            loaded["config"],
            loaded["tokenizer"],
            loaded["model"],
            loaded.get("architecture"),
            loaded.get("adapter"),
        )
    config, tokenizer, model = loaded
    return config, tokenizer, model, None, None


def _parameter_bytes(model):
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        values = parameters()
    else:
        named_parameters = getattr(model, "named_parameters", None)
        if not callable(named_parameters):
            raise IncompleteRun(
                "INCOMPLETE_MODEL_LOAD",
                "loaded model exposes no parameter iterator",
            )
        values = (
            parameter for _, parameter in named_parameters()
        )
    return sum(
        int(parameter.numel() * parameter.element_size())
        for parameter in values
    )


def main(
    argv=None,
    *,
    reference_loader=load_official_reference,
    adapter_factory=None,
    cuda_module=None,
    stdout=None,
):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inspect-model", "run-canonical"))
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--contract-sha256", required=True)
    arguments = parser.parse_args(argv)
    output = stdout or sys.stdout
    _require_contract_sha256(arguments.contract_sha256)
    cuda_api = cuda_module or torch.cuda
    lifecycle_memory = [
        capture_memory_snapshot(
            snapshot_id="lifecycle:before_model_load",
            phase="before_model_load",
            request_id="__model__",
            request_generation=0,
            components=[],
            cuda=cuda_api,
        )
    ]
    loaded = reference_loader(Path(arguments.model_dir))
    lifecycle_memory.append(capture_memory_snapshot(
        snapshot_id="lifecycle:after_model_load",
        phase="after_model_load",
        request_id="__model__",
        request_generation=0,
        components=[],
        cuda=cuda_api,
    ))
    config, tokenizer, model, architecture, loaded_adapter = (
        _loaded_reference_parts(loaded)
    )
    architecture = architecture or inspect_model(
        model=model,
        config=config,
        tokenizer=tokenizer,
    )
    write_json_atomic(
        Path(arguments.run_dir) / "architecture.json",
        architecture,
    )
    if arguments.command == "inspect-model":
        output.write(json.dumps(architecture, sort_keys=True) + "\n")
        return 0
    parameter_bytes = _parameter_bytes(model)
    fp32_adapter_factory = None
    if adapter_factory is None:
        if loaded_adapter is None:
            loaded_adapter = ReferenceStateAdapter(
                model=model,
                layer_schedule=architecture["layer_schedule"],
                vocab_size=architecture["tokenizer_vocab_size"],
                model_vocab_size=architecture["model_vocab_size"],
                device="cuda:0",
                requested_model_dtype="bfloat16",
                parameter_dtypes=architecture["parameter_dtypes"],
            )
        adapter_holder = [loaded_adapter]

        def adapter_factory():
            if not adapter_holder:
                raise IncompleteRun(
                    "INCOMPLETE_REFERENCE_SEMANTICS",
                    "BF16 adapter factory was reused",
                )
            return adapter_holder.pop()

        def fp32_adapter_factory():
            fp32_loaded = reference_loader(
                Path(arguments.model_dir),
                requested_dtype="float32",
            )
            (
                _fp32_config,
                _fp32_tokenizer,
                _fp32_model,
                fp32_architecture,
                fp32_adapter,
            ) = _loaded_reference_parts(fp32_loaded)
            fp32_architecture = fp32_architecture or inspect_model(
                model=_fp32_model,
                config=_fp32_config,
                tokenizer=_fp32_tokenizer,
            )
            if (
                _architecture_identity(fp32_architecture)
                != _architecture_identity(architecture)
            ):
                raise IncompleteRun(
                    "INCOMPLETE_ARCHITECTURE",
                    "FP32 control architecture differs from BF16",
                )
            if fp32_adapter is None:
                fp32_adapter = ReferenceStateAdapter(
                    model=_fp32_model,
                    layer_schedule=architecture["layer_schedule"],
                    vocab_size=architecture["tokenizer_vocab_size"],
                    model_vocab_size=architecture["model_vocab_size"],
                    device="cuda:0",
                    requested_model_dtype="float32",
                    parameter_dtypes=architecture["parameter_dtypes"],
                )
            return fp32_adapter
        loaded_adapter = None
        loaded = None
        model = None
        gc.collect()
    result = run_reference_case_matrix(
        adapter_factory=adapter_factory,
        fp32_adapter_factory=fp32_adapter_factory,
        architecture=architecture,
        run_dir=Path(arguments.run_dir),
        contract_sha256=arguments.contract_sha256,
        parameter_bytes=parameter_bytes,
        cuda_module=cuda_api,
    )
    parameter_bytes = result["summary"]["parameter_bytes"]
    del adapter_factory
    del loaded_adapter
    del loaded
    del model
    gc.collect()
    lifecycle_memory.append(capture_memory_snapshot(
        snapshot_id="lifecycle:after_model_release",
        phase="after_model_release",
        request_id="__model__",
        request_generation=0,
        components=[],
        cuda=cuda_api,
    ))
    result["memory_snapshots"] = [
        lifecycle_memory[0],
        lifecycle_memory[1],
        *result["memory_snapshots"],
        lifecycle_memory[2],
    ]
    result["summary"] = emit_raw_probe_artifacts(
        run_dir=Path(arguments.run_dir),
        architecture=architecture,
        case_rows=result["case_rows"],
        state_snapshots=result["state_snapshots"],
        state_components=result["state_components"],
        memory_snapshots=result["memory_snapshots"],
        parameter_bytes=parameter_bytes,
        max_memory_allocated=max(
            int(result["summary"]["max_memory_allocated"]),
            int(cuda_api.max_memory_allocated()),
        ),
        max_memory_reserved=max(
            int(result["summary"]["max_memory_reserved"]),
            int(cuda_api.max_memory_reserved()),
        ),
        dtype_profiles=result["dtype_profiles"],
    )
    output.write(json.dumps(result["summary"], sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
