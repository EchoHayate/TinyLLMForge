import copy
import hashlib
import re
import time

from pilot_protocol import FLAGS, require
from tokenizer_measurement import _state, measure
from validation_contract import digest


def generate_candidate(model, tokenizer, messages, *, expected_tokenizer_sha256):
    """Generate one fixed greedy candidate without loading, executing or scoring it.

    CUDA peaks include the resident model and cover generation and decoding after
    input/model validation. CPU memory measurements are unavailable, not zero.
    The caller owns model integrity, exclusive access and failed-call accounting.
    """
    started = time.perf_counter_ns()
    require(type(expected_tokenizer_sha256) is str
            and re.fullmatch(r"[0-9a-f]{64}", expected_tokenizer_sha256) is not None,
            "invalid expected tokenizer SHA256")
    copied_messages = copy.deepcopy(messages)
    messages_sha256 = digest(copied_messages)
    measured = measure(tokenizer, copied_messages, answer=None)
    require(measured["state_sha256"] == expected_tokenizer_sha256,
            "tokenizer SHA256 mismatch")
    require(measured["status"] == "OK" and measured["input"]["input_tokens"] <= 4096,
            "input token limit exceeded")
    state, vocabulary = _state(tokenizer)
    require(digest(state) == expected_tokenizer_sha256, "tokenizer state changed")
    require(digest(messages) == messages_sha256
            and digest(copied_messages) == messages_sha256, "messages changed")

    import torch

    require(not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda"),
            "generation requires disabled autocast")
    require(isinstance(model, torch.nn.Module), "torch model required")
    require(all(not module.training for module in model.modules()),
            "caller model must be eval")
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    tensors = parameters + buffers
    require(bool(tensors), "empty model state")
    require(all(not parameter.requires_grad and parameter.grad is None
                for parameter in parameters), "caller model must be frozen without gradients")
    require(all(not tensor.requires_grad for tensor in buffers),
            "buffer gradients unsupported")
    require(all(not tensor.is_inference() for tensor in tensors),
            "inference tensor state unsupported")
    require(all(tensor.layout == torch.strided for tensor in tensors),
            "dense model state required")
    float_dtypes = (torch.float32, torch.bfloat16)
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    require(all(parameter.dtype in float_dtypes for parameter in parameters),
            "BF16 or FP32 parameters required")
    require(all(tensor.dtype in float_dtypes + integer_dtypes for tensor in buffers),
            "unsupported buffer dtype")
    devices = {tensor.device for tensor in tensors}
    require(len(devices) == 1, "single model device required")
    device = next(iter(devices))
    require(device.type in ("cpu", "cuda"), "CPU or CUDA model required")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    prefix = list(measured["input"]["input_ids"])
    output_ids = []
    stop_reason = "length"
    synchronize()
    preparation_ended = time.perf_counter_ns()
    generation_started = time.perf_counter_ns()
    with torch.no_grad():
        for _ in range(256):
            inputs = torch.tensor([prefix], dtype=torch.long, device=device)
            output = model(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                use_cache=False, return_dict=True)
            logits = getattr(output, "logits", None)
            require(isinstance(logits, torch.Tensor) and logits.is_floating_point()
                    and logits.layout == torch.strided, "floating dense logits required")
            require(logits.ndim == 3 and tuple(logits.shape[:2]) == (1, len(prefix))
                    and logits.shape[2] > 0, "logits shape mismatch")
            require(logits.device == device, "logits device mismatch")
            last_logits = logits[0, -1]
            require(bool(torch.isfinite(last_logits).all()), "nonfinite last-position logits")
            selected = int(torch.argmax(last_logits).item())
            require(selected in vocabulary, "selected token outside tokenizer vocabulary")
            output_ids.append(selected)
            del output, logits, last_logits, inputs
            if selected == state["eos_token_id"]:
                stop_reason = "eos"
                break
            prefix.append(selected)
    synchronize()
    generation_ended = time.perf_counter_ns()
    decoding_started = time.perf_counter_ns()
    decode_ids = output_ids[:-1] if stop_reason == "eos" else list(output_ids)
    candidate = tokenizer.decode(
        decode_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    require(type(candidate) is str, "decoded candidate must be a string")
    candidate_bytes = candidate.encode("utf-8")
    require(len(candidate_bytes) <= 65536, "decoded candidate UTF-8 byte limit exceeded")
    synchronize()
    decoding_ended = time.perf_counter_ns()
    require(digest(_state(tokenizer)[0]) == expected_tokenizer_sha256,
            "tokenizer state changed")
    require(digest(messages) == messages_sha256
            and digest(copied_messages) == messages_sha256, "messages changed")
    result = {
        "kind": "LORA_GREEDY_CANDIDATE_UNATTESTED", "device_type": device.type,
        "tokenizer_state_sha256": expected_tokenizer_sha256,
        "messages_sha256": messages_sha256,
        "prompt_sha256": measured["input"]["prompt_sha256"],
        "input_tokens": measured["input"]["input_tokens"],
        "output_tokens": len(output_ids), "output_ids": output_ids,
        "candidate": candidate,
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "stop_reason": stop_reason, "model_calls": 1, "forward_calls": len(output_ids),
        "timings_ns": {
            "preparation": preparation_ended - started,
            "generation": generation_ended - generation_started,
            "decoding": decoding_ended - decoding_started,
        },
        "peak_allocated_bytes": None, "peak_reserved_bytes": None,
        **{flag: False for flag in FLAGS},
    }
    synchronize()
    if device.type == "cuda":
        result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    result["timings_ns"]["wall"] = time.perf_counter_ns() - started
    return result
