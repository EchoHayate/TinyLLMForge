import copy
from contextlib import contextmanager
import os
import time

from pilot_protocol import FLAGS, require


def run_roundtrip(model, dataset, artifact_path, *, emit=None):
    """Qualify one fixed adapter roundtrip without generation or scoring.

    Mount timings include the enclosed probe and mount/unmount integrity checks.
    Nested probe messages do not add duplicate timings. CUDA peaks include the
    resident base model; CPU peaks are unavailable, not zero.
    """
    started = time.perf_counter_ns()
    if os.path.lexists(artifact_path):
        raise FileExistsError("adapter artifact already exists")
    require(emit is None or callable(emit), "callable emit required")

    from training_inputs import validate_training

    dataset = copy.deepcopy(dataset)
    validate_training(dataset)

    import torch

    from adapter_artifact import load_adapter, mounted_adapter, save_adapter
    from training_worker import train_adapter

    require(torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),
            "roundtrip requires enabled autograd without inference mode")
    require(not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda"),
            "roundtrip requires disabled autocast")
    require(isinstance(model, torch.nn.Module), "torch model required")
    require(all(not module.training for module in model.modules()),
            "caller model must be eval")
    require(all(not parameter.requires_grad and parameter.grad is None
                for parameter in model.parameters()), "caller model must be frozen")
    tensors = list(model.parameters()) + list(model.buffers())
    require(bool(tensors), "empty model")
    require(all(not tensor.is_inference() for tensor in tensors),
            "inference-tensor base state is unsupported")
    devices = {tensor.device for tensor in tensors}
    require(len(devices) == 1, "single model device required")
    device = next(iter(devices))
    require(device.type in ("cpu", "cuda"), "CPU or CUDA required")
    timings = {}

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @contextmanager
    def phase(name, *, timed=True):
        if emit is not None:
            emit({"phase": name, "event": "start"})
        if timed:
            synchronize()
        phase_started = time.perf_counter_ns()
        status = "error"
        try:
            yield
            if timed:
                synchronize()
            status = "ok"
        finally:
            if timed:
                timings[name] = time.perf_counter_ns() - phase_started
            if emit is not None:
                emit({"phase": name, "event": "end", "status": status})

    row = dataset["rows"][0]
    prompt_tokens = row["prompt_tokens"]

    def probe(active):
        with torch.no_grad():
            inputs = torch.tensor([row["input_ids"][:prompt_tokens]],
                                  dtype=torch.long, device=device)
            mask = torch.tensor([row["attention_mask"][:prompt_tokens]],
                                dtype=torch.long, device=device)
            output = active(input_ids=inputs, attention_mask=mask,
                            use_cache=False, return_dict=True)
            logits = output.logits[:, -1, :].detach().cpu().clone()
            require(bool(torch.isfinite(logits).all()), "nonfinite probe logits")
            return logits

    def mounted_probe(result, name, *, state="adapter"):
        with phase(f"{name}_mount"):
            with mounted_adapter(model, result, state=state) as active:
                with phase(f"{name}_probe", timed=False):
                    return probe(active)

    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with phase("base_probe"):
        base = probe(model)
    with phase("training"):
        trained = train_adapter(model, dataset)
    initial = mounted_probe(trained, "initial_adapter", state="initial_adapter")
    checks = {"initial_equals_base": torch.equal(initial, base)}
    require(checks["initial_equals_base"], "initial_equals_base failed")
    final = mounted_probe(trained, "final_adapter")
    with phase("save_adapter"):
        artifact = save_adapter(artifact_path, trained)
    with phase("load_adapter"):
        loaded = load_adapter(artifact_path, expected_sha256=artifact["sha256"])
    reloaded = mounted_probe(loaded["result"], "reloaded_adapter")
    checks["reload_equals_final"] = torch.equal(reloaded, final)
    require(checks["reload_equals_final"], "reload_equals_final failed")
    with phase("restored_probe"):
        restored = probe(model)
    checks["restored_equals_base"] = torch.equal(restored, base)
    require(checks["restored_equals_base"], "restored_equals_base failed")
    synchronize()
    peak_allocated = None
    peak_reserved = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    timings["roundtrip_wall"] = time.perf_counter_ns() - started
    return {
        "training_receipt": trained["receipt"],
        "checks": checks,
        "artifact": artifact,
        "timings_ns": timings,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        **{flag: False for flag in FLAGS},
    }
