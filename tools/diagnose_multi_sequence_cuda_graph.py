#!/usr/bin/env python3
"""Run one isolated multi-sequence CUDA Graph diagnostic case."""

from __future__ import annotations

import argparse
import atexit
import importlib.util
import json
import os
import platform
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "tools" / "multi_sequence_cuda_graph_contract.py"
RAGGED_TARGETS = (
    32,
    64,
    96,
    128,
    192,
    224,
    255,
    257,
    288,
    320,
    384,
    448,
    512,
    576,
    640,
    704,
)
TENSOR_SHARD_REQUIRED_FIELDS = (
    "schema_version",
    "case_id",
    "step_ids",
    "row_ids",
    "dtype",
    "shape",
    "tensor",
)


def _load_contract():
    spec = importlib.util.spec_from_file_location(
        "multi_sequence_cuda_graph_contract",
        CONTRACT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract()


@dataclass
class CapturedDecodeGraph:
    graph_size: int
    graph: object
    input_ids: object
    positions: object
    slot_mapping: object
    context_lens: object
    block_tables: object
    outputs: object
    layer_outputs: object


def repeat_to_exact_token_count(seed: list[int], target_count: int) -> list[int]:
    if not seed:
        raise ValueError("seed tokens must not be empty")
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    repetitions = (target_count + len(seed) - 1) // len(seed)
    return (list(seed) * repetitions)[:target_count]


def build_prompt_plan(tokenizer, batch_size: int) -> dict[str, list[list[int]]]:
    if batch_size <= 0 or batch_size > len(RAGGED_TARGETS):
        raise ValueError(f"unsupported prompt-plan batch size: {batch_size}")
    short = tokenizer.encode("CUDA graph row isolation test.")
    long_seed = tokenizer.encode(
        "Explain why deterministic batching metadata matters for KV-cache "
        "decode. "
    )
    ragged = [
        repeat_to_exact_token_count(long_seed, RAGGED_TARGETS[index])
        for index in range(batch_size)
    ]
    distinct = [
        tokenizer.encode(f"Distinct CUDA graph row {index}.")
        for index in range(batch_size)
    ]
    if batch_size >= 2:
        distinct[1] = list(distinct[0])
    return {
        "uniform-short": [list(short) for _ in range(batch_size)],
        "ragged-context": ragged,
        "duplicate-and-distinct": distinct,
    }


def build_kv_observation_plan(
    *,
    active_slots,
    graph_size: int,
    inactive_slots,
    total_slots: int,
) -> dict:
    active = [int(slot) for slot in active_slots]
    inactive = [int(slot) for slot in inactive_slots]
    if graph_size < len(active):
        raise ValueError("graph_size is smaller than active slot count")
    if total_slots <= 0:
        raise ValueError("total_slots must be positive")
    observed = set(active)
    observed.update(inactive)
    observed.add(0)
    for slot in observed:
        if slot < 0 or slot >= total_slots:
            raise ValueError(f"KV slot out of range: {slot}")

    candidates = [
        total_slots - 1,
        total_slots // 2,
        total_slots // 3,
        (2 * total_slots) // 3,
        total_slots // 4,
        (3 * total_slots) // 4,
    ]
    sentinel_slots = []
    for candidate in candidates:
        if candidate not in observed and candidate not in sentinel_slots:
            sentinel_slots.append(candidate)
        if len(sentinel_slots) == 3:
            break
    if len(sentinel_slots) < 3:
        for candidate in range(total_slots):
            if candidate not in observed and candidate not in sentinel_slots:
                sentinel_slots.append(candidate)
            if len(sentinel_slots) == 3:
                break
    if len(sentinel_slots) < 3:
        raise ValueError("insufficient untouched KV sentinel slots")
    return {
        "active_write_slots": active,
        "slot_zero": 0,
        "inactive_declared_slots": inactive,
        "sentinel_slots": sentinel_slots,
    }


def build_step_row(
    *,
    observed_argmax_token_ids,
    reference_next_input_token_ids,
    **extra,
) -> dict:
    observed = [int(token_id) for token_id in observed_argmax_token_ids]
    reference = [
        int(token_id) for token_id in reference_next_input_token_ids
    ]
    if len(observed) != len(reference):
        raise ValueError("observed and reference token rows must match")
    return {
        **extra,
        "observed_argmax_token_ids": observed,
        "reference_next_input_token_ids": reference,
        "teacher_forcing_diverged": observed != reference,
    }


def validate_tensor_shard(shard: dict) -> None:
    for field in TENSOR_SHARD_REQUIRED_FIELDS:
        if field not in shard:
            raise ValueError(f"tensor shard missing {field}")
    if shard["schema_version"] != 1:
        raise ValueError("tensor shard schema_version must be 1")
    if not isinstance(shard["case_id"], str) or not shard["case_id"]:
        raise ValueError("tensor shard case_id must be non-empty")
    if not isinstance(shard["dtype"], str) or not shard["dtype"]:
        raise ValueError("tensor shard dtype must be non-empty")
    if not isinstance(shard["shape"], list):
        raise ValueError("tensor shard shape must be a list")
    if shard["step_ids"] != list(range(len(shard["step_ids"]))):
        raise ValueError("tensor shard step_ids must be ordered from zero")
    if shard["row_ids"] != list(range(len(shard["row_ids"]))):
        raise ValueError("tensor shard row_ids must be ordered from zero")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_bytes(path, contract.canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    payload = b"".join(
        contract.canonical_json_bytes(row) + b"\n" for row in rows
    )
    _atomic_write_bytes(path, payload)


def _atomic_torch_save(path: Path, value: object) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(value, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _artifact_record(*, output_dir: Path, path: Path) -> dict:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "sha256": contract.sha256_file(path),
    }


def _load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_case(case_spec: dict):
    required = {
        "batch_size",
        "trajectory",
        "mode",
        "repetition",
        "graph_size",
    }
    missing = sorted(required - set(case_spec))
    if missing:
        raise ValueError(f"case spec missing fields: {missing}")
    case = contract.DiagnosticCase(
        batch_size=int(case_spec["batch_size"]),
        trajectory=str(case_spec["trajectory"]),
        mode=str(case_spec["mode"]),
        repetition=int(case_spec["repetition"]),
        graph_size=int(case_spec["graph_size"]),
    )
    expected = {candidate.case_id: candidate for candidate in (
        contract.build_diagnostic_matrix()
    )}
    if case.case_id not in expected or case != expected[case.case_id]:
        raise ValueError(f"case is outside frozen diagnostic matrix: {case}")
    return case


def _load_reference_tokens(path: Path, case) -> list[list[int]] | None:
    if case.mode == "eager":
        return None
    value = _load_json(path)
    if not isinstance(value, list):
        raise ValueError("reference token array must be a list")
    expected_steps = contract.WARMUP_STEPS + contract.MEASURED_STEPS
    if len(value) != expected_steps:
        raise ValueError(
            f"reference token array must have {expected_steps} steps"
        )
    normalized = []
    for step_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != case.batch_size:
            raise ValueError(
                "reference token row shape mismatch at "
                f"step {step_index}"
            )
        normalized.append([int(token_id) for token_id in row])
    return normalized


def _validate_model_preflight(model_path: Path) -> dict:
    from transformers import AutoConfig

    if not model_path.is_dir():
        raise ValueError(f"model directory does not exist: {model_path}")
    hf_config = AutoConfig.from_pretrained(str(model_path))
    if getattr(hf_config, "model_type", None) != "qwen3":
        raise ValueError(
            f"diagnostic requires Qwen3, got {hf_config.model_type!r}"
        )
    dtype = str(getattr(hf_config, "torch_dtype", ""))
    if "bfloat16" not in dtype:
        raise ValueError(f"diagnostic requires BF16, got {dtype!r}")
    return {
        "model_type": hf_config.model_type,
        "torch_dtype": dtype,
        "num_hidden_layers": int(hf_config.num_hidden_layers),
        "hidden_size": int(hf_config.hidden_size),
    }


def _engine_kwargs() -> dict:
    return {
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "max_num_seqs": 32,
        "max_num_batched_tokens": 32768,
        "max_model_len": 1024,
        "gpu_memory_utilization": 0.55,
        "quantization": None,
        "cpu_offload": False,
        "quest_top_k_blocks": -1,
        "kv_cartridge_blocks": 0,
        "am_compact_blocks": 0,
        "max_num_prefill_tokens_per_step": 0,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_adaptive_mixed": False,
        "chunked_prefill_slo_mixed": False,
        "kv_quant_bits": 0,
        "kv_offload_mvp0": False,
        "act_quant_bits": 0,
        "smoothquant_scale_path": None,
    }


def _validate_runtime_config(llm, case) -> None:
    import torch

    config = llm.model_runner.config
    if config.tensor_parallel_size != 1:
        raise ValueError("diagnostic requires TP=1")
    if config.max_num_seqs < 32:
        raise ValueError("diagnostic requires max_num_seqs >= 32")
    if config.hf_config.torch_dtype != torch.bfloat16:
        raise ValueError("diagnostic requires actual BF16 model dtype")
    expected_graph_size = contract.diagnostic_graph_size(
        case.batch_size,
        case.mode,
    )
    if case.graph_size != expected_graph_size:
        raise ValueError("case graph size does not match frozen contract")
    unsupported = {
        "quantization": config.quantization is not None,
        "cpu_offload": config.cpu_offload,
        "quest_top_k_blocks": config.quest_top_k_blocks > 0,
        "kv_cartridge_blocks": config.kv_cartridge_blocks > 0,
        "am_compact_blocks": config.am_compact_blocks > 0,
        "chunked_prefill_mixed_batch": config.chunked_prefill_mixed_batch,
        "chunked_prefill_adaptive_mixed": (
            config.chunked_prefill_adaptive_mixed
        ),
        "chunked_prefill_slo_mixed": config.chunked_prefill_slo_mixed,
        "kv_quant_bits": config.kv_quant_bits != 0,
        "kv_offload_mvp0": config.kv_offload_mvp0,
        "act_quant_bits": config.act_quant_bits != 0,
        "smoothquant_scale_path": (
            config.smoothquant_scale_path is not None
        ),
    }
    active = sorted(name for name, enabled in unsupported.items() if enabled)
    if active:
        raise ValueError(f"excluded features enabled: {active}")


def _prefill_live_sequences(llm, prompts: list[list[int]]):
    from tinyvllm.sampling_params import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=(
            contract.WARMUP_STEPS + contract.MEASURED_STEPS + 1
        ),
        ignore_eos=True,
    )
    for prompt in prompts:
        llm.add_request(list(prompt), sampling_params)
    outputs, num_tokens = llm.step()
    if outputs or num_tokens <= 0:
        raise RuntimeError("expected one complete prefill step")
    seqs = list(llm.scheduler.running)
    if len(seqs) != len(prompts):
        raise RuntimeError("prefill did not retain every live sequence")
    return seqs


def _reserve_decode_slots(llm, seqs) -> None:
    manager = llm.scheduler.block_manager
    for seq in seqs:
        if not manager.can_append(seq):
            raise RuntimeError(
                f"insufficient KV capacity for sequence {seq.seq_id}"
            )
        manager.may_append(seq)


def _restore_kv_slots(runner, physical_slots: list[int], snapshot) -> None:
    import torch

    block_ids = torch.tensor(
        [slot // runner.block_size for slot in physical_slots],
        device=runner.kv_cache.device,
        dtype=torch.long,
    )
    offsets = torch.tensor(
        [slot % runner.block_size for slot in physical_slots],
        device=runner.kv_cache.device,
        dtype=torch.long,
    )
    runner.kv_cache[0, :, block_ids, offsets].copy_(
        snapshot["keys"].to(runner.kv_cache.device)
    )
    runner.kv_cache[1, :, block_ids, offsets].copy_(
        snapshot["values"].to(runner.kv_cache.device)
    )
    torch.cuda.synchronize()


def _register_layer_hooks(layers, layer_outputs):
    handles = []
    graph_size = layer_outputs.size(2)
    for layer_index, layer in enumerate(layers):
        def capture_output(module, inputs, output, index=layer_index):
            del module, inputs
            layer_outputs[0, index, :graph_size].copy_(
                output[0][:graph_size]
            )
            layer_outputs[1, index, :graph_size].copy_(
                output[1][:graph_size]
            )

        handles.append(layer.register_forward_hook(capture_output))
    return handles


def _forward_without_autograd(model, input_ids, positions):
    import torch

    with torch.inference_mode():
        return model(input_ids, positions)


def _forward_and_logits_without_autograd(model, input_ids, positions):
    import torch

    with torch.inference_mode():
        hidden_states = model(input_ids, positions)
        return model.compute_logits(hidden_states)


def _compute_logits_without_autograd(model, hidden_states):
    import torch

    with torch.inference_mode():
        return model.compute_logits(hidden_states)


def _capture_decode_graph(runner, case, dynamic_context):
    import torch

    from tinyvllm.utils.context import reset_context, set_context

    graph_size = case.graph_size
    config = runner.config
    device = runner.kv_cache.device
    max_blocks = (
        config.max_model_len + runner.block_size - 1
    ) // runner.block_size
    input_ids = torch.zeros(
        graph_size,
        dtype=torch.int64,
        device=device,
    )
    positions = torch.zeros(
        graph_size,
        dtype=torch.int64,
        device=device,
    )
    slot_mapping = torch.zeros(
        graph_size,
        dtype=torch.int32,
        device=device,
    )
    context_lens = torch.zeros(
        graph_size,
        dtype=torch.int32,
        device=device,
    )
    block_tables = torch.zeros(
        graph_size,
        max_blocks,
        dtype=torch.int32,
        device=device,
    )
    outputs = torch.zeros(
        graph_size,
        config.hf_config.hidden_size,
        dtype=config.hf_config.torch_dtype,
        device=device,
    )
    layer_outputs = torch.zeros(
        2,
        config.hf_config.num_hidden_layers,
        graph_size,
        config.hf_config.hidden_size,
        dtype=config.hf_config.torch_dtype,
        device=device,
    )

    active_size = case.batch_size
    input_ids[:active_size].copy_(dynamic_context["input_ids"])
    positions[:active_size].copy_(dynamic_context["positions"])
    slot_mapping[:active_size].copy_(dynamic_context["slot_mapping"])
    context_lens[:active_size].copy_(dynamic_context["context_lens"])
    active_columns = dynamic_context["block_tables"].size(1)
    block_tables[:active_size, :active_columns].copy_(
        dynamic_context["block_tables"]
    )

    capture_slots = sorted(
        set(int(slot) for slot in slot_mapping.tolist())
    )
    capture_snapshot = runner.snapshot_kv_slots(capture_slots)
    layers = runner.model.model.layers
    handles = _register_layer_hooks(layers, layer_outputs)
    graph = torch.cuda.CUDAGraph()
    try:
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        outputs.copy_(
            _forward_without_autograd(
                runner.model,
                input_ids,
                positions,
            )
        )
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            outputs.copy_(
                _forward_without_autograd(
                    runner.model,
                    input_ids,
                    positions,
                )
            )
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
        reset_context()
        _restore_kv_slots(runner, capture_slots, capture_snapshot)
    return CapturedDecodeGraph(
        graph_size=graph_size,
        graph=graph,
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        outputs=outputs,
        layer_outputs=layer_outputs,
    )


def _dynamic_decode_inputs(runner, seqs) -> dict:
    from tinyvllm.utils.context import get_context

    input_ids, positions = runner.prepare_decode(seqs)
    context = get_context()
    return {
        "input_ids": input_ids,
        "positions": positions,
        "slot_mapping": context.slot_mapping,
        "context_lens": context.context_lens,
        "block_tables": context.block_tables,
    }


def _run_eager_step(runner, dynamic):
    import torch

    layers = runner.model.model.layers
    layer_outputs = torch.zeros(
        2,
        runner.config.hf_config.num_hidden_layers,
        dynamic["input_ids"].size(0),
        runner.config.hf_config.hidden_size,
        dtype=runner.config.hf_config.torch_dtype,
        device=runner.kv_cache.device,
    )
    handles = _register_layer_hooks(layers, layer_outputs)
    try:
        logits = _forward_and_logits_without_autograd(
            runner.model,
            dynamic["input_ids"],
            dynamic["positions"],
        )
        torch.cuda.synchronize()
        return (
            logits.detach().cpu().contiguous(),
            layer_outputs.detach().cpu().contiguous(),
        )
    finally:
        for handle in handles:
            handle.remove()


def _run_graph_step(runner, captured, dynamic, active_size: int):
    import torch

    from tinyvllm.utils.context import set_context

    for tensor in (
        captured.input_ids,
        captured.positions,
        captured.slot_mapping,
        captured.context_lens,
        captured.block_tables,
        captured.outputs,
        captured.layer_outputs,
    ):
        tensor.zero_()
    captured.input_ids[:active_size].copy_(dynamic["input_ids"])
    captured.positions[:active_size].copy_(dynamic["positions"])
    captured.slot_mapping[:active_size].copy_(dynamic["slot_mapping"])
    captured.context_lens[:active_size].copy_(dynamic["context_lens"])
    active_columns = dynamic["block_tables"].size(1)
    captured.block_tables[:active_size, :active_columns].copy_(
        dynamic["block_tables"]
    )
    set_context(
        False,
        slot_mapping=captured.slot_mapping,
        context_lens=captured.context_lens,
        block_tables=captured.block_tables,
    )
    captured.graph.replay()
    logits = _compute_logits_without_autograd(
        runner.model,
        captured.outputs[:active_size],
    )
    torch.cuda.synchronize()
    return (
        logits.detach().cpu().contiguous(),
        captured.layer_outputs[
            :, :, :active_size
        ].detach().cpu().contiguous(),
    )


def _ordered_observation_slots(plan: dict) -> list[int]:
    ordered = []
    for slot in (
        plan["active_write_slots"]
        + [plan["slot_zero"]]
        + plan["inactive_declared_slots"]
        + plan["sentinel_slots"]
    ):
        if slot not in ordered:
            ordered.append(slot)
    return ordered


def _kv_step_payload(
    *,
    plan: dict,
    before: dict,
    after: dict,
) -> dict:
    return {
        "plan": plan,
        "slot_ids": _ordered_observation_slots(plan),
        "keys_before": before["keys"],
        "values_before": before["values"],
        "keys_after": after["keys"],
        "values_after": after["values"],
    }


def _save_tensor_shard(
    *,
    output_dir: Path,
    path: Path,
    case,
    tensor,
) -> dict:
    shard = {
        "schema_version": 1,
        "case_id": case.case_id,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "step_ids": list(range(contract.MEASURED_STEPS)),
        "row_ids": list(range(case.batch_size)),
        "tensor": tensor.cpu().contiguous(),
    }
    validate_tensor_shard(shard)
    _atomic_torch_save(path, shard)
    return {
        **_artifact_record(output_dir=output_dir, path=path),
        "metadata": contract.tensor_metadata(shard["tensor"]),
    }


def _save_kv_shard(
    *,
    output_dir: Path,
    path: Path,
    case,
    step_payloads: list[dict],
) -> dict:
    import torch

    shard = {
        "schema_version": 1,
        "case_id": case.case_id,
        "step_ids": list(range(contract.MEASURED_STEPS)),
        "row_ids": list(range(case.batch_size)),
        "slot_ids": [payload["slot_ids"] for payload in step_payloads],
        "plans": [payload["plan"] for payload in step_payloads],
        "keys_before": torch.stack(
            [payload["keys_before"] for payload in step_payloads]
        ),
        "values_before": torch.stack(
            [payload["values_before"] for payload in step_payloads]
        ),
        "keys_after": torch.stack(
            [payload["keys_after"] for payload in step_payloads]
        ),
        "values_after": torch.stack(
            [payload["values_after"] for payload in step_payloads]
        ),
    }
    _atomic_torch_save(path, shard)
    return {
        **_artifact_record(output_dir=output_dir, path=path),
        "shape": {
            key: list(shard[key].shape)
            for key in (
                "keys_before",
                "values_before",
                "keys_after",
                "values_after",
            )
        },
    }


def _process_environment() -> dict:
    return {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tinyvllm_dist_port": os.environ.get("TINYVLLM_DIST_PORT"),
        "master_port": os.environ.get("MASTER_PORT"),
    }


def run_case(
    *,
    model_path: Path,
    case_spec_path: Path,
    reference_tokens_path: Path,
    output_dir: Path,
) -> dict:
    import torch

    from tinyvllm import LLM
    from tinyvllm.utils.context import reset_context

    output_dir.mkdir(parents=True, exist_ok=True)
    case = _parse_case(_load_json(case_spec_path))
    reference_tokens = _load_reference_tokens(reference_tokens_path, case)
    model_preflight = _validate_model_preflight(model_path)
    _atomic_write_json(
        output_dir / "process_environment.json",
        {
            **_process_environment(),
            "model_preflight": model_preflight,
            "case": asdict(case),
        },
    )

    llm = None
    try:
        llm = LLM(str(model_path), **_engine_kwargs())
        _validate_runtime_config(llm, case)
        prompts = build_prompt_plan(
            tokenizer=llm.tokenizer,
            batch_size=case.batch_size,
        )[case.trajectory]
        seqs = _prefill_live_sequences(llm, prompts)
        runner = llm.model_runner

        _reserve_decode_slots(llm, seqs)
        first_dynamic = _dynamic_decode_inputs(runner, seqs)
        captured = None
        if case.mode != "eager":
            captured = _capture_decode_graph(
                runner,
                case,
                first_dynamic,
            )
        reset_context()

        total_steps = contract.WARMUP_STEPS + contract.MEASURED_STEPS
        raw_rows = []
        layer_rows = []
        kv_rows = []
        logits_steps = []
        layer_steps = []
        kv_step_payloads = []
        observed_reference_tokens = []

        for absolute_step in range(total_steps):
            if absolute_step > 0:
                _reserve_decode_slots(llm, seqs)
            dynamic = _dynamic_decode_inputs(runner, seqs)
            active_slots = [
                int(slot) for slot in dynamic["slot_mapping"].tolist()
            ]
            inactive_slots = (
                [0] if case.graph_size > case.batch_size else []
            )
            observation_plan = build_kv_observation_plan(
                active_slots=active_slots,
                graph_size=case.graph_size,
                inactive_slots=inactive_slots,
                total_slots=(
                    runner.kv_cache.size(2) * runner.block_size
                ),
            )
            observation_slots = _ordered_observation_slots(
                observation_plan
            )
            before = runner.snapshot_kv_slots(observation_slots)

            if case.mode == "eager":
                logits, layers = _run_eager_step(runner, dynamic)
            else:
                logits, layers = _run_graph_step(
                    runner,
                    captured,
                    dynamic,
                    case.batch_size,
                )
            after = runner.snapshot_kv_slots(observation_slots)
            observed_tokens = [
                int(token_id)
                for token_id in torch.argmax(logits, dim=-1).tolist()
            ]
            if reference_tokens is None:
                reference_row = list(observed_tokens)
                observed_reference_tokens.append(reference_row)
            else:
                reference_row = reference_tokens[absolute_step]
            for seq, token_id in zip(seqs, reference_row):
                seq.append_token(int(token_id))

            if absolute_step >= contract.WARMUP_STEPS:
                measured_step = absolute_step - contract.WARMUP_STEPS
                raw_rows.append(
                    build_step_row(
                        case_id=case.case_id,
                        step_id=measured_step,
                        absolute_step=absolute_step,
                        mode=case.mode,
                        batch_size=case.batch_size,
                        graph_size=case.graph_size,
                        active_write_slots=active_slots,
                        observed_argmax_token_ids=observed_tokens,
                        reference_next_input_token_ids=reference_row,
                    )
                )
                layer_rows.append(
                    {
                        "case_id": case.case_id,
                        "step_id": measured_step,
                        "required_layer_count": (
                            runner.config.hf_config.num_hidden_layers
                        ),
                        "observed_layer_count": layers.size(1),
                        "shape": list(layers.shape),
                        "finite": bool(torch.isfinite(layers).all().item()),
                    }
                )
                kv_rows.append(
                    {
                        "case_id": case.case_id,
                        "step_id": measured_step,
                        **observation_plan,
                        "observed_slot_ids": observation_slots,
                    }
                )
                logits_steps.append(logits)
                layer_steps.append(layers)
                kv_step_payloads.append(
                    _kv_step_payload(
                        plan=observation_plan,
                        before=before,
                        after=after,
                    )
                )
            reset_context()

        if case.mode == "eager":
            _atomic_write_json(
                reference_tokens_path,
                observed_reference_tokens,
            )

        logits_tensor = torch.stack(logits_steps)
        layers_tensor = torch.stack(layer_steps)
        artifact_records = {}
        artifact_records["logits"] = _save_tensor_shard(
            output_dir=output_dir,
            path=(
                output_dir
                / "tensors"
                / "logits"
                / f"{case.case_id}.pt"
            ),
            case=case,
            tensor=logits_tensor,
        )
        artifact_records["layers"] = _save_tensor_shard(
            output_dir=output_dir,
            path=(
                output_dir
                / "tensors"
                / "layers"
                / f"{case.case_id}.pt"
            ),
            case=case,
            tensor=layers_tensor,
        )
        artifact_records["kv"] = _save_kv_shard(
            output_dir=output_dir,
            path=(
                output_dir
                / "tensors"
                / "kv"
                / f"{case.case_id}.pt"
            ),
            case=case,
            step_payloads=kv_step_payloads,
        )
        _write_jsonl(output_dir / "raw_rows.jsonl", raw_rows)
        _write_jsonl(
            output_dir / "layer_observations.jsonl",
            layer_rows,
        )
        _write_jsonl(output_dir / "kv_observations.jsonl", kv_rows)
        for name in (
            "raw_rows.jsonl",
            "layer_observations.jsonl",
            "kv_observations.jsonl",
            "process_environment.json",
        ):
            path = output_dir / name
            artifact_records[name] = _artifact_record(
                output_dir=output_dir,
                path=path,
            )
        result = {
            "schema_version": 1,
            "status": "PASS",
            "case": asdict(case),
            "case_id": case.case_id,
            "artifacts": artifact_records,
            "measured_steps": contract.MEASURED_STEPS,
            "warmup_steps": contract.WARMUP_STEPS,
        }
        _atomic_write_json(output_dir / "case_result.json", result)
        _atomic_write_bytes(output_dir / "exitcode", b"0\n")
        return result
    finally:
        reset_context()
        if llm is not None:
            atexit.unregister(llm.exit)
            llm.exit()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one isolated multi-sequence CUDA Graph case",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--case-spec", type=Path, required=True)
    parser.add_argument("--reference-tokens", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_case(
            model_path=args.model,
            case_spec_path=args.case_spec,
            reference_tokens_path=args.reference_tokens,
            output_dir=output_dir,
        )
        return 0
    except Exception as exc:
        traceback_text = traceback.format_exc()
        failure = {
            "schema_version": 1,
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback_text.splitlines()[-40:],
        }
        _atomic_write_json(output_dir / "case_result.json", failure)
        _atomic_write_bytes(output_dir / "stderr.txt", traceback_text.encode())
        _atomic_write_bytes(output_dir / "exitcode", b"1\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
