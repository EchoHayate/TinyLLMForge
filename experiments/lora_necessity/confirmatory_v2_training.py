import copy
import re
import time

from confirmatory_v2_contract import (
    CONFIRMATORY_FILE_PINS,
    SUPERVISION_IDS,
    confirmatory_training_jobs,
)
from pilot_protocol import FLAGS, configuration, fields, require
from tokenizer_measurement import measure
from validation_contract import digest


DATASET_FIELDS = (
    "kind",
    "version",
    "supervision_sha256",
    "tokenizer_state_sha256",
    "eos_token_id",
    "pad_token_id",
    "rows",
    "dataset_sha256",
)
ROW_FIELDS = (
    "task_id",
    "input_ids",
    "attention_mask",
    "labels",
    "prompt_tokens",
    "target_tokens",
    "messages_sha256",
    "answer_sha256",
)


def _check_sha256(value):
    require(
        type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None,
        "invalid SHA256",
    )


def validate_confirmatory_training(dataset):
    fields(dataset, DATASET_FIELDS)
    require(
        dataset["kind"] == "LORA_CONFIRMATORY_V2_TRAINING_INPUTS"
        and type(dataset["version"]) is int
        and dataset["version"] == 1,
        "training dataset header",
    )
    for sha_name in ("supervision_sha256", "tokenizer_state_sha256", "dataset_sha256"):
        _check_sha256(dataset[sha_name])
    require(
        dataset["supervision_sha256"] == CONFIRMATORY_FILE_PINS["supervision.json"],
        "supervision.json pin mismatch",
    )
    for tok_field in ("eos_token_id", "pad_token_id"):
        require(type(dataset[tok_field]) is int and dataset[tok_field] >= 0, "special token ID")

    rows = dataset["rows"]
    require(type(rows) is list and len(rows) == len(SUPERVISION_IDS), "8 training rows required")
    for row, expected_tid in zip(rows, SUPERVISION_IDS):
        fields(row, ROW_FIELDS)
        require(row["task_id"] == expected_tid, "supervision task order")
        _check_sha256(row["messages_sha256"])
        _check_sha256(row["answer_sha256"])
        p_toks = row["prompt_tokens"]
        t_toks = row["target_tokens"]
        require(
            type(p_toks) is int and 0 < p_toks <= 4096 and type(t_toks) is int and 0 < t_toks <= 256,
            "training row token limits",
        )
        total_len = p_toks + t_toks
        for seq_name in ("input_ids", "attention_mask", "labels"):
            seq = row[seq_name]
            require(
                type(seq) is list and len(seq) == total_len and all(type(val) is int for val in seq),
                "training integer sequence",
            )
        require(all(val >= 0 for val in row["input_ids"]), "nonnegative input_ids")
        require(all(val == 1 for val in row["attention_mask"]), "all-ones attention_mask")
        target_slice = row["input_ids"][p_toks:]
        require(
            target_slice[-1] == dataset["eos_token_id"]
            and target_slice.count(dataset["eos_token_id"]) == 1,
            "unique terminal EOS in target",
        )
        require(
            row["labels"] == ([-100] * p_toks) + target_slice,
            "answer-only label alignment",
        )

    expected_sha = digest({key_n: val_v for key_n, val_v in dataset.items() if key_n != "dataset_sha256"})
    require(dataset["dataset_sha256"] == expected_sha, "training dataset_sha256 mismatch")
    return expected_sha


def prepare_confirmatory_training(tokenizer, supervision_raw):
    jobs = confirmatory_training_jobs(supervision_raw)
    require(tuple(job["task_id"] for job in jobs) == SUPERVISION_IDS, "ordered V01..V08 jobs")
    rows = []
    state_sha256 = None
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    for job in jobs:
        measured = measure(tokenizer, job["messages"], job["answer"])
        require(measured["status"] == "OK", "training row exceeded token budget")
        if state_sha256 is None:
            state_sha256 = measured["state_sha256"]
        require(
            measured["state_sha256"] == state_sha256
            and tokenizer.eos_token_id == eos_id
            and tokenizer.pad_token_id == pad_id,
            "tokenizer state changed during training prep",
        )
        p_rec = measured["input"]
        t_rec = measured["target"]
        input_ids = list(p_rec["input_ids"]) + list(t_rec["target_ids"])
        rows.append({
            "task_id": job["task_id"],
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": ([-100] * p_rec["input_tokens"]) + list(t_rec["target_ids"]),
            "prompt_tokens": p_rec["input_tokens"],
            "target_tokens": t_rec["target_tokens"],
            "messages_sha256": p_rec["messages_sha256"],
            "answer_sha256": t_rec["answer_sha256"],
        })

    dataset = {
        "kind": "LORA_CONFIRMATORY_V2_TRAINING_INPUTS",
        "version": 1,
        "supervision_sha256": CONFIRMATORY_FILE_PINS["supervision.json"],
        "tokenizer_state_sha256": state_sha256,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "rows": rows,
    }
    dataset["dataset_sha256"] = digest(dataset)
    validate_confirmatory_training(dataset)
    return dataset


def train_confirmatory_adapter(model, dataset):
    import torch
    from torch import nn
    from training_worker import (
        LoRAProjection,
        _adapter_state,
        _finite,
        _optimizer_parameters,
        _state,
        _synchronize,
        _targets,
        _tensor_digest,
        answer_loss,
    )

    started = time.perf_counter_ns()
    require(
        torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),
        "training requires enabled autograd",
    )
    require(
        not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda"),
        "training requires disabled autocast",
    )
    copied_ds = copy.deepcopy(dataset)
    dataset_sha256 = validate_confirmatory_training(copied_ds)
    require(isinstance(model, nn.Module), "torch nn.Module required")
    base_state = _state(model)
    require(bool(base_state), "nonempty model state required")
    devices = {tensor.device for tensor in base_state.values()}
    require(len(devices) == 1, "single model device required")
    device = next(iter(devices))
    require(device.type in ("cpu", "cuda"), "CPU or CUDA device required")
    require(all(tensor.layout == torch.strided for tensor in base_state.values()), "dense base state")
    require(
        all(param.dtype == torch.bfloat16 and param.grad is None for param in model.parameters()),
        "BF16 base parameters without grad required",
    )
    require(
        all(not buf.is_floating_point() or buf.dtype in (torch.float32, torch.bfloat16) for buf in model.buffers()),
        "supported buffer dtype",
    )
    _finite(base_state.values())
    targets = _targets(model)
    base_sha256 = _tensor_digest(base_state)
    train_cfg = configuration()["training"]

    modes = [(mod, mod.training) for mod in model.modules()]
    grad_flags = [(param, param.requires_grad) for param in model.parameters()]
    wrappers = []
    originals = []
    steps_log = []
    initial_adapter = None
    final_adapter = None
    opt_bytes = None
    failure = None
    peak_alloc = None
    peak_resv = None

    rng_devices = [device.index or 0] if device.type == "cuda" else []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    with torch.random.fork_rng(devices=rng_devices):
        torch.default_generator.manual_seed(train_cfg["seed"])
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(train_cfg["seed"])
        try:
            model.requires_grad_(False)
            model.eval()
            for path_str, base_lin in targets:
                parent_path, attr_name = path_str.rsplit(".", 1)
                parent_mod = model.get_submodule(parent_path)
                wrap_mod = LoRAProjection(base_lin)
                originals.append((parent_mod, attr_name, base_lin))
                setattr(parent_mod, attr_name, wrap_mod)
                wrappers.append((path_str, wrap_mod))

            trainable_params = [
                param_obj
                for _p, wrap_mod in wrappers
                for param_obj in (wrap_mod.adapter_a, wrap_mod.adapter_b)
            ]
            require(
                {id(param_obj) for param_obj in model.parameters() if param_obj.requires_grad}
                == {id(param_obj) for param_obj in trainable_params},
                "trainable parameter allowlist",
            )
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=train_cfg["learning_rate"],
                weight_decay=train_cfg["weight_decay"],
            )
            _optimizer_parameters(optimizer, trainable_params)
            initial_adapter = _adapter_state(wrappers)
            _synchronize(device)
            train_t0 = time.perf_counter_ns()
            prep_ns = train_t0 - started

            for step_idx in range(train_cfg["steps"]):
                row = copied_ds["rows"][step_idx % 8]
                step_t0 = time.perf_counter_ns()
                inp_t = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
                msk_t = torch.tensor([row["attention_mask"]], dtype=torch.long, device=device)
                lbl_t = torch.tensor([row["labels"]], dtype=torch.long, device=device)
                _optimizer_parameters(optimizer, trainable_params)
                optimizer.zero_grad(set_to_none=True)
                out_obj = model(input_ids=inp_t, attention_mask=msk_t, use_cache=False, return_dict=True)
                loss_val = answer_loss(out_obj.logits, lbl_t)
                _finite([loss_val])
                loss_val.backward()
                require(all(param_obj.grad is not None for param_obj in trainable_params), "missing grad")
                _finite(param_obj.grad for param_obj in trainable_params)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, train_cfg["gradient_clip"], error_if_nonfinite=True
                )
                _finite([grad_norm])
                optimizer.step()
                _finite(trainable_params)
                opt_tensors = [
                    val_t
                    for state_dict in optimizer.state.values()
                    for val_t in state_dict.values()
                    if isinstance(val_t, torch.Tensor)
                ]
                _finite(opt_tensors)
                _synchronize(device)
                steps_log.append({
                    "step": step_idx + 1,
                    "task_id": row["task_id"],
                    "loss": float(loss_val.item()),
                    "gradient_norm": float(grad_norm.item()),
                    "elapsed_ns": time.perf_counter_ns() - step_t0,
                })
                del inp_t, msk_t, lbl_t, out_obj, loss_val, grad_norm

            _synchronize(device)
            training_ns = time.perf_counter_ns() - train_t0
            verify_t0 = time.perf_counter_ns()
            final_adapter = _adapter_state(wrappers)
            opt_bytes = sum(val_t.numel() * val_t.element_size() for val_t in opt_tensors)
            if device.type == "cuda":
                peak_alloc = int(torch.cuda.max_memory_allocated(device))
                peak_resv = int(torch.cuda.max_memory_reserved(device))
        except BaseException as err:
            failure = err
            raise
        finally:
            for parent_mod, attr_name, orig_lin in reversed(originals):
                setattr(parent_mod, attr_name, orig_lin)
            for param_obj, req_grad in grad_flags:
                param_obj.requires_grad_(req_grad)
            for mod_obj, was_train in modes:
                mod_obj.training = was_train
            restored_sha = _tensor_digest(_state(model))
            if restored_sha != base_sha256:
                raise ValueError("base state mutated during LoRA training") from failure

    init_ad_sha = _tensor_digest(initial_adapter)
    fin_ad_sha = _tensor_digest(final_adapter)
    require(fin_ad_sha != init_ad_sha, "adapter weights did not update")
    _synchronize(device)
    finished = time.perf_counter_ns()
    receipt = {
        "kind": "LORA_TRAINING_CORE_RESULT_UNATTESTED",
        "dataset_sha256": dataset_sha256,
        "configuration_sha256": digest(configuration()),
        "tokenizer_state_sha256": copied_ds["tokenizer_state_sha256"],
        "base_sha256": base_sha256,
        "target_paths": [path_str for path_str, _lin in targets],
        "initial_adapter_sha256": init_ad_sha,
        "adapter_sha256": fin_ad_sha,
        "device_type": device.type,
        "steps": steps_log,
        "supervised_tokens": sum(row["target_tokens"] for row in copied_ds["rows"]) * 2,
        "preparation_ns": prep_ns,
        "training_ns": training_ns,
        "verification_ns": finished - verify_t0,
        "core_wall_ns": finished - started,
        "adapter_bytes": sum(ten.numel() * ten.element_size() for ten in final_adapter.values()),
        "optimizer_tensor_bytes": opt_bytes,
        "model_load_ns": None,
        "adapter_save_ns": None,
        "launch_ns": None,
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_resv,
        "end_to_end_ns": None,
        **{flag: False for flag in FLAGS},
    }
    return {"receipt": receipt, "initial_adapter": initial_adapter, "adapter": final_adapter}
