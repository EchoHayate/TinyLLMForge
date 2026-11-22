import copy
import hashlib
import io
import math
import os
from pathlib import Path
import re
import time
from contextlib import contextmanager

import torch
from torch import nn

from pilot_protocol import FLAGS, configuration, fields, require
from training_worker import LoRAProjection, _adapter_state, _finite, _state, _targets, _tensor_digest
from validation_contract import digest


ARTIFACT_KIND = "LORA_ADAPTER_ARTIFACT_UNATTESTED"


def _json_metadata(value):
    if type(value) is dict:
        require(all(type(name) is str for name in value), "JSON string keys required")
        for item in value.values():
            _json_metadata(item)
    elif type(value) is list:
        for item in value:
            _json_metadata(item)
    else:
        require(value is None or type(value) in (str, int, float, bool),
                "JSON-native receipt metadata required")
        require(type(value) is not float or math.isfinite(value), "nonfinite receipt metadata")


def _sha256(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "invalid SHA256 digest")


def _validate_result(result):
    fields(result, ("receipt", "initial_adapter", "adapter"))
    receipt = result["receipt"]
    require(type(receipt) is dict, "receipt required")
    _json_metadata(receipt)
    require(receipt.get("kind") == "LORA_TRAINING_CORE_RESULT_UNATTESTED", "receipt kind")
    for name in ("dataset_sha256", "configuration_sha256", "tokenizer_state_sha256",
                 "base_sha256", "initial_adapter_sha256", "adapter_sha256"):
        _sha256(receipt.get(name))
    require(receipt["configuration_sha256"] == digest(configuration()), "configuration mismatch")
    require(all(receipt.get(flag) is False for flag in FLAGS), "false admission flags required")
    paths = receipt.get("target_paths")
    require(type(paths) is list and bool(paths) and len(paths) % 2 == 0,
            "complete target paths required")
    require(paths == [
        f"model.layers.{index}.self_attn.{projection}"
        for index in range(len(paths) // 2) for projection in ("q_proj", "v_proj")],
        "consecutive q/v target paths required")
    names = {f"{path}.{suffix}" for path in paths for suffix in ("adapter_a", "adapter_b")}
    require(type(receipt.get("adapter_bytes")) is int and receipt["adapter_bytes"] > 0,
            "adapter byte count required")
    for state in ("initial_adapter", "adapter"):
        tensors = result[state]
        fields(tensors, names)
        for name, tensor in tensors.items():
            require(type(tensor) is torch.Tensor and tensor.layout == torch.strided
                    and tensor.device.type == "cpu" and tensor.dtype == torch.float32
                    and not tensor.requires_grad and tensor.grad is None,
                    "dense CPU FP32 adapter tensors without gradients required")
            require(tensor.ndim == 2 and all(size > 0 for size in tensor.shape),
                    "positive two-dimensional adapter shape required")
            rank_axis = 0 if name.endswith(".adapter_a") else 1
            require(tensor.shape[rank_axis] == 8, "adapter rank shape mismatch")
            if state == "adapter":
                require(tensor.shape == result["initial_adapter"][name].shape,
                        "initial/final shape mismatch")
            elif name.endswith(".adapter_b"):
                require(bool((tensor == 0).all()), "initial B must be zero")
        _finite(tensors.values())
        require(sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
                == receipt["adapter_bytes"], "adapter byte count mismatch")
        require(_tensor_digest(tensors) == receipt[f"{state}_sha256"],
                "adapter digest mismatch")
    require(receipt["initial_adapter_sha256"] != receipt["adapter_sha256"],
            "no adapter update observed")
    return receipt


def save_adapter(path, result):
    """Exclusively save an unattested snapshot; failed writes remain evidence."""
    started = time.perf_counter_ns()
    _validate_result(result)
    snapshot = {
        "receipt": copy.deepcopy(result["receipt"]),
        **{state: {name: tensor.detach().clone() for name, tensor in result[state].items()}
           for state in ("initial_adapter", "adapter")},
    }
    _validate_result(snapshot)
    stream = io.BytesIO()
    torch.save({"kind": ARTIFACT_KIND, "version": 1, "result": snapshot}, stream)
    raw = stream.getvalue()
    sha256 = hashlib.sha256(raw).hexdigest()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        written = os.write(descriptor, raw)
        if written != len(raw):
            raise OSError("incomplete adapter write; preserve file and discard artifact")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"sha256": sha256, "file_bytes": len(raw),
            "save_ns": time.perf_counter_ns() - started}


def load_adapter(path, *, expected_sha256):
    """Load checked bytes with weights-only deserialization, never an unsafe fallback."""
    started = time.perf_counter_ns()
    _sha256(expected_sha256)
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "file digest mismatch")
    envelope = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    fields(envelope, ("kind", "version", "result"))
    require(envelope["kind"] == ARTIFACT_KIND, "artifact kind")
    require(type(envelope["version"]) is int and envelope["version"] == 1, "artifact version")
    _validate_result(envelope["result"])
    return {"result": envelope["result"], "load_ns": time.perf_counter_ns() - started}


@contextmanager
def mounted_adapter(model, result, *, state="adapter"):
    """Mount for single-owner inference with fresh caller-owned KV state per request.

    CPU use is synthetic only. Integrity errors require discarding the indicated
    model or mount; mutated base bytes are detected, not rolled back.
    Inference-tensor base state is unsupported, even inside inference_mode.
    """
    require(type(state) is str and state in ("adapter", "initial_adapter"),
            "adapter or initial_adapter state required")
    require(not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda"),
            "mounting requires disabled autocast")
    receipt = _validate_result(result)
    require(isinstance(model, nn.Module), "torch model required")
    base_state = _state(model)
    require(bool(base_state), "empty model")
    require(all(not tensor.is_inference() for tensor in base_state.values()),
            "inference-tensor base state is unsupported")
    devices = {tensor.device for tensor in base_state.values()}
    require(len(devices) == 1, "single model device required")
    device = next(iter(devices))
    require(device.type in ("cpu", "cuda"), "CPU or CUDA required")
    require(all(tensor.layout == torch.strided for tensor in base_state.values()),
            "dense base state required")
    require(all(parameter.dtype == torch.bfloat16 and parameter.grad is None
                for parameter in model.parameters()), "BF16 base with no existing gradients required")
    require(all(not tensor.is_floating_point()
                or tensor.dtype in (torch.float32, torch.bfloat16)
                for tensor in model.buffers()), "unsupported base buffer dtype")
    _finite(base_state.values())
    targets = _targets(model)
    require([name for name, _ in targets] == receipt["target_paths"], "target paths mismatch")
    base_sha256 = _tensor_digest(base_state)
    require(base_sha256 == receipt["base_sha256"], "base digest mismatch")
    selected = result[state]
    selected_sha256 = receipt[f"{state}_sha256"]
    for path, base in targets:
        require(tuple(base.weight.shape) == (base.out_features, base.in_features)
                and (base.bias is None or tuple(base.bias.shape) == (base.out_features,)),
                "base projection dimensions mismatch")
        require(tuple(selected[f"{path}.adapter_a"].shape) == (8, base.in_features)
                and tuple(selected[f"{path}.adapter_b"].shape) == (base.out_features, 8),
                "adapter projection dimensions mismatch")
    modes = [(module, module.training) for module in model.modules()]
    grad_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    wrappers = []
    originals = []
    installed = False
    failure = None
    rng_devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
        try:
            model.requires_grad_(False)
            model.eval()
            for path, base in targets:
                parent_path, attribute = path.rsplit(".", 1)
                parent = model.get_submodule(parent_path)
                wrapper = LoRAProjection(base)
                wrapper.adapter_a.copy_(selected[f"{path}.adapter_a"])
                wrapper.adapter_b.copy_(selected[f"{path}.adapter_b"])
                wrapper.requires_grad_(False)
                wrapper.eval()
                originals.append((parent, attribute, base))
                setattr(parent, attribute, wrapper)
                wrappers.append((path, wrapper))
            installed = True
            yield model
        except BaseException as error:
            failure = error
            raise
        finally:
            adapter_error = None
            try:
                if installed:
                    require(all(model.get_submodule(path) is wrapper
                                and wrapper.base is base for (path, wrapper), (_, base)
                                in zip(wrappers, targets)), "mounted adapter structure changed")
                    require(all(not parameter.requires_grad and parameter.grad is None
                                for _, wrapper in wrappers for parameter in
                                (wrapper.adapter_a, wrapper.adapter_b)),
                            "mounted adapter frozen state changed")
                    require(_tensor_digest(_adapter_state(wrappers)) == selected_sha256,
                            "mounted adapter bytes changed")
            except BaseException as error:
                adapter_error = error
            restoration_errors = []
            base_grad_changed = any(parameter.grad is not None for parameter, _ in grad_flags)
            restorations = (
                list(reversed(originals))
                + [(parameter, attribute, value) for parameter, requires_grad in grad_flags
                   for attribute, value in (("grad", None), ("requires_grad", requires_grad))]
                + [(module, "training", training) for module, training in modes])
            for owner, attribute, value in restorations:
                try:
                    if attribute == "requires_grad":
                        owner.requires_grad_(value)
                    else:
                        setattr(owner, attribute, value)
                except BaseException as error:
                    restoration_errors.append(error)
            base_error = None
            try:
                restored_state = _state(model)
                restored_sha256 = _tensor_digest(restored_state)
                same_state = (set(restored_state) == set(base_state)
                              and all(restored_state[name] is tensor and tensor.device == device
                                      for name, tensor in base_state.items()))
            except BaseException as integrity_error:
                base_error = "base state unverifiable"
                restoration_errors.append(integrity_error)
            else:
                if restored_sha256 != base_sha256 or not same_state or base_grad_changed:
                    base_error = "base state changed"
            if restoration_errors or base_error is not None:
                message = base_error or "base state restoration failed"
                details = "; ".join(
                    f"{type(error).__name__}: {error}" for error in restoration_errors)
                raise ValueError(f"{message}; discard model; {details}") from (
                    failure if failure is not None else
                    restoration_errors[0] if restoration_errors else None)
            if adapter_error is not None:
                raise ValueError("mounted adapter changed or unverifiable; discard mount") from (
                    failure if failure is not None else adapter_error)
