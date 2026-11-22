import copy
import hashlib
import math
import time
from collections import Counter

import torch
from torch import nn
from torch.nn import functional

from pilot_protocol import FLAGS, configuration, require
from validation_contract import digest


class LoRAProjection(nn.Module):
    def __init__(self, base):
        super().__init__()
        require(type(base) is nn.Linear, "ordinary linear projection required")
        self.base = base
        base.requires_grad_(False)
        self.adapter_a = nn.Parameter(torch.empty(
            8, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.adapter_b = nn.Parameter(torch.zeros(
            base.out_features, 8, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.adapter_a, a=math.sqrt(5))

    def forward(self, inputs):
        baseline = self.base(inputs)
        update = functional.linear(functional.linear(
            inputs.float(), self.adapter_a), self.adapter_b)
        return baseline + (2 * update).to(baseline.dtype)


def answer_loss(logits, labels):
    require(isinstance(logits, torch.Tensor) and logits.ndim == 3,
            "batched causal logits required")
    require(isinstance(labels, torch.Tensor) and labels.dtype == torch.long
            and tuple(labels.shape) == tuple(logits.shape[:2]),
            "aligned integer labels required")
    require(logits.shape[1] >= 2 and labels.device == logits.device,
            "causal sequence/device mismatch")
    shifted = labels[:, 1:]
    selected = shifted != -100
    require(bool(selected.any()), "answer targets required")
    targets = shifted[selected]
    require(bool(((targets >= 0) & (targets < logits.shape[-1])).all()),
            "target outside vocabulary")
    return functional.cross_entropy(logits[:, :-1][selected].float(), targets)


def _state(model):
    return dict(
        [(f"parameter:{name}", value) for name, value in
         model.named_parameters(remove_duplicate=False)]
        + [(f"buffer:{name}", value) for name, value in
           model.named_buffers(remove_duplicate=False)])


def _tensor_digest(tensors):
    records = {}
    for name, tensor in sorted(tensors.items()):
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        records[name] = {
            "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return digest(records)


def _finite(tensors):
    require(all(bool(torch.isfinite(tensor).all()) for tensor in tensors),
            "nonfinite tensor")


def _targets(model):
    config = getattr(model, "config", None)
    count = getattr(config, "num_hidden_layers", None)
    require(getattr(config, "model_type", None) == "qwen3"
            and type(count) is int and count > 0, "Qwen3 decoder layout required")
    expected = [
        f"model.layers.{index}.self_attn.{name}"
        for index in range(count) for name in ("q_proj", "v_proj")]
    observed = [(name, module) for name, module in
                model.named_modules(remove_duplicate=False)
                if name.rsplit(".", 1)[-1] in ("q_proj", "v_proj")]
    require([name for name, _ in observed] == expected, "target layout mismatch")
    require(all(type(module) is nn.Linear for _, module in observed),
            "ordinary nonadapted targets required")
    storage_counts = Counter(
        parameter.untyped_storage().data_ptr() for _, parameter in
        model.named_parameters(remove_duplicate=False))
    require(all(storage_counts[parameter.untyped_storage().data_ptr()] == 1
                for _, module in observed for parameter in module.parameters()),
            "shared target parameter storage")
    return observed


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _adapter_state(wrappers):
    return {f"{path}.{name}": parameter.detach().cpu().clone()
            for path, wrapper in wrappers for name, parameter in
            (("adapter_a", wrapper.adapter_a), ("adapter_b", wrapper.adapter_b))}


def _optimizer_parameters(optimizer, parameters):
    observed = [parameter for group in optimizer.param_groups
                for parameter in group["params"]]
    require([id(parameter) for parameter in observed]
            == [id(parameter) for parameter in parameters], "optimizer parameter allowlist")


def train_adapter(model, dataset):
    from training_inputs import validate_training

    started = time.perf_counter_ns()
    require(torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),
            "training requires enabled autograd")
    require(not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda"),
            "training requires disabled autocast")
    dataset = copy.deepcopy(dataset)
    dataset_sha256 = validate_training(dataset)
    require(isinstance(model, nn.Module), "torch model required")
    state = _state(model)
    require(bool(state), "empty model")
    devices = {tensor.device for tensor in state.values()}
    require(len(devices) == 1, "single model device required")
    device = next(iter(devices))
    require(device.type in ("cpu", "cuda"), "CPU or CUDA required")
    require(all(tensor.layout == torch.strided for tensor in state.values()),
            "dense base state required")
    require(all(parameter.dtype == torch.bfloat16 and parameter.grad is None
                for parameter in model.parameters()), "BF16 base with no existing gradients required")
    require(all(not tensor.is_floating_point()
                or tensor.dtype in (torch.float32, torch.bfloat16)
                for tensor in model.buffers()), "unsupported base buffer dtype")
    _finite(state.values())
    targets = _targets(model)
    base_sha256 = _tensor_digest(state)
    train_config = configuration()["training"]
    modes = [(module, module.training) for module in model.modules()]
    grad_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    wrappers = []
    originals = []
    steps = []
    initial_adapter = None
    adapter = None
    optimizer_tensor_bytes = None
    failure = None
    rng_devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices):
        torch.default_generator.manual_seed(train_config["seed"])
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(train_config["seed"])
        try:
            model.requires_grad_(False)
            model.eval()
            for path, base in targets:
                parent_path, attribute = path.rsplit(".", 1)
                parent = model.get_submodule(parent_path)
                wrapper = LoRAProjection(base)
                originals.append((parent, attribute, base))
                setattr(parent, attribute, wrapper)
                wrappers.append((path, wrapper))
            parameters = [parameter for _, wrapper in wrappers
                          for parameter in (wrapper.adapter_a, wrapper.adapter_b)]
            require({id(parameter) for parameter in model.parameters()
                     if parameter.requires_grad} == {id(parameter) for parameter in parameters},
                    "trainable parameter allowlist")
            optimizer = torch.optim.AdamW(
                parameters, lr=train_config["learning_rate"],
                weight_decay=train_config["weight_decay"])
            _optimizer_parameters(optimizer, parameters)
            initial_adapter = _adapter_state(wrappers)
            _synchronize(device)
            training_started = time.perf_counter_ns()
            preparation_ns = training_started - started
            for index in range(train_config["steps"]):
                row = dataset["rows"][index % 8]
                step_started = time.perf_counter_ns()
                inputs = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
                mask = torch.tensor([row["attention_mask"]], dtype=torch.long, device=device)
                labels = torch.tensor([row["labels"]], dtype=torch.long, device=device)
                _optimizer_parameters(optimizer, parameters)
                optimizer.zero_grad(set_to_none=True)
                output = model(input_ids=inputs, attention_mask=mask,
                               use_cache=False, return_dict=True)
                loss = answer_loss(output.logits, labels)
                _finite([loss])
                loss.backward()
                require(all(parameter.grad is not None for parameter in parameters),
                        "missing adapter gradient")
                _finite(parameter.grad for parameter in parameters)
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, train_config["gradient_clip"], error_if_nonfinite=True)
                _finite([norm])
                optimizer.step()
                _finite(parameters)
                optimizer_tensors = [
                    value for item in optimizer.state.values() for value in item.values()
                    if isinstance(value, torch.Tensor)]
                _finite(optimizer_tensors)
                _synchronize(device)
                steps.append({
                    "step": index + 1, "task_id": row["task_id"],
                    "loss": loss.item(), "gradient_norm": norm.item(),
                    "elapsed_ns": time.perf_counter_ns() - step_started,
                })
            _synchronize(device)
            training_ns = time.perf_counter_ns() - training_started
            verification_started = time.perf_counter_ns()
            adapter = _adapter_state(wrappers)
            optimizer_tensor_bytes = sum(
                tensor.numel() * tensor.element_size() for tensor in optimizer_tensors)
        except BaseException as error:
            failure = error
            raise
        finally:
            for parent, attribute, original in reversed(originals):
                setattr(parent, attribute, original)
            for parameter, requires_grad in grad_flags:
                parameter.requires_grad_(requires_grad)
            for module, training in modes:
                module.training = training
            try:
                restored_sha256 = _tensor_digest(_state(model))
            except BaseException as integrity_error:
                raise ValueError("base state unverifiable; discard model") from (
                    failure if failure is not None else integrity_error)
            if restored_sha256 != base_sha256:
                raise ValueError("base state changed; discard model") from failure
    initial_sha256 = _tensor_digest(initial_adapter)
    final_sha256 = _tensor_digest(adapter)
    require(final_sha256 != initial_sha256, "no adapter update observed")
    _synchronize(device)
    finished = time.perf_counter_ns()
    receipt = {
        "kind": "LORA_TRAINING_CORE_RESULT_UNATTESTED",
        "dataset_sha256": dataset_sha256,
        "configuration_sha256": digest(configuration()),
        "tokenizer_state_sha256": dataset["tokenizer_state_sha256"],
        "base_sha256": base_sha256, "target_paths": [name for name, _ in targets],
        "initial_adapter_sha256": initial_sha256, "adapter_sha256": final_sha256,
        "device_type": device.type, "steps": steps,
        "supervised_tokens": sum(row["target_tokens"] for row in dataset["rows"]) * 2,
        "preparation_ns": preparation_ns, "training_ns": training_ns,
        "verification_ns": finished - verification_started,
        "core_wall_ns": finished - started,
        "adapter_bytes": sum(tensor.numel() * tensor.element_size()
                             for tensor in adapter.values()),
        "optimizer_tensor_bytes": optimizer_tensor_bytes,
        "model_load_ns": None, "adapter_save_ns": None, "launch_ns": None,
        "peak_allocated_bytes": None, "peak_reserved_bytes": None, "end_to_end_ns": None,
        **{flag: False for flag in FLAGS},
    }
    return {"receipt": receipt, "initial_adapter": initial_adapter, "adapter": adapter}
