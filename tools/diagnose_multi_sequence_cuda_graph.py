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
SPLIT_POLICY_PATH = (
    ROOT / "tinyvllm" / "engine" / "flash_attn_split_policy.py"
)
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


def _load_split_policy():
    spec = importlib.util.spec_from_file_location(
        "flash_attn_split_policy",
        SPLIT_POLICY_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


split_policy = _load_split_policy()


@dataclass(frozen=True)
class StepSplitPolicy:
    inputs: split_policy.FlashAttentionSplitInputs
    identity: split_policy.FlashAttentionGraphIdentity

    @property
    def effective_num_splits(self) -> int:
        return self.identity.effective_num_splits


def step_policy_evidence(policy: StepSplitPolicy) -> dict:
    inputs = policy.inputs
    identity = policy.identity
    return {
        "split_policy_name": contract.HEURISTIC_POLICY_NAME,
        "flash_attn_version": identity.flash_attn_version,
        "page_table_width": identity.page_table_width,
        "effective_num_splits": identity.effective_num_splits,
        "heuristic_batch_size": inputs.batch_size,
        "heuristic_num_query_heads": inputs.num_query_heads,
        "heuristic_num_kv_heads": inputs.num_kv_heads,
        "heuristic_head_dim": inputs.head_dim,
        "heuristic_page_block_size": inputs.page_block_size,
        "heuristic_max_seqlen_q": inputs.max_seqlen_q,
        "heuristic_multi_processor_count": (
            inputs.multi_processor_count
        ),
        "graph_batch_size": identity.graph_batch_size,
        "graph_identity_sha256": identity.sha256,
    }


def build_step_policy_rows(
    policy: StepSplitPolicy,
    *,
    raw: dict,
    layer: dict,
    kv: dict,
) -> dict[str, dict]:
    evidence = step_policy_evidence(policy)
    return {
        "raw": {**raw, **evidence},
        "layer": {**layer, **evidence},
        "kv": {**kv, **evidence},
    }


def graph_identity_summary(identities) -> list[dict]:
    ordered = []
    seen = set()
    for identity in identities:
        if identity in seen:
            continue
        seen.add(identity)
        ordered.append(
            {
                "sha256": identity.sha256,
                "page_table_width": identity.page_table_width,
                "effective_num_splits": identity.effective_num_splits,
                "graph_batch_size": identity.graph_batch_size,
            }
        )
    return ordered


@dataclass
class CapturedDecodeGraph:
    identity: split_policy.FlashAttentionGraphIdentity
    graph: object
    input_ids: object
    positions: object
    slot_mapping: object
    context_lens: object
    block_tables: object
    outputs: object
    layer_outputs: object


def _required_int_attribute(owner, field: str) -> int:
    try:
        value = getattr(owner, field)
    except AttributeError as exc:
        raise ValueError(f"missing {field}") from exc
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if normalized <= 0:
        raise ValueError(f"{field} must be positive")
    return normalized


def build_step_split_policy(
    *,
    runner,
    dynamic_context: dict,
    active_batch_size: int,
    graph_batch_size: int,
    flash_attn_version: str,
) -> StepSplitPolicy:
    if flash_attn_version != split_policy.FLASH_ATTN_VERSION:
        raise ValueError(
            "step split policy requires FlashAttention 2.6.3"
        )
    try:
        hf_config = runner.config.hf_config
    except AttributeError as exc:
        raise ValueError("missing runner.config.hf_config") from exc
    try:
        block_tables = dynamic_context["block_tables"]
    except (KeyError, TypeError) as exc:
        raise ValueError("missing dynamic block_tables") from exc
    try:
        page_table_width = int(block_tables.size(1))
    except (AttributeError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("invalid block_tables page-table width") from exc
    try:
        kv_cache_device = runner.kv_cache.device
    except AttributeError as exc:
        raise ValueError("missing runner.kv_cache.device") from exc

    import torch

    device_properties = torch.cuda.get_device_properties(kv_cache_device)
    multi_processor_count = _required_int_attribute(
        device_properties,
        "multi_processor_count",
    )
    inputs = split_policy.FlashAttentionSplitInputs(
        batch_size=int(active_batch_size),
        num_query_heads=_required_int_attribute(
            hf_config,
            "num_attention_heads",
        ),
        num_kv_heads=_required_int_attribute(
            hf_config,
            "num_key_value_heads",
        ),
        head_dim=_required_int_attribute(hf_config, "head_dim"),
        page_block_size=_required_int_attribute(runner, "block_size"),
        page_table_width=page_table_width,
        max_seqlen_q=1,
        multi_processor_count=multi_processor_count,
    )
    identity = split_policy.build_flash_attn_263_graph_identity(
        graph_batch_size=int(graph_batch_size),
        inputs=inputs,
        flash_attn_version=flash_attn_version,
    )
    return StepSplitPolicy(inputs=inputs, identity=identity)


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


def _all_frozen_cases():
    return (
        tuple(contract.build_diagnostic_matrix())
        + tuple(contract.build_legacy_compatibility_matrix())
    )


def _parse_case(case_spec: dict):
    case_id = str(case_spec.get("case_id", ""))
    expected = {case.case_id: case for case in _all_frozen_cases()}
    if case_id not in expected:
        raise ValueError(f"case is outside frozen matrices: {case_id}")
    case = expected[case_id]
    if case_spec != {"case_id": case.case_id, **asdict(case)}:
        raise ValueError(f"case identity drift: {case.case_id}")
    return case


def execution_split_count(case) -> int:
    if case.flash_attn_num_splits is None:
        raise ValueError(
            "heuristic cases derive flash_attn_num_splits per step"
        )
    return int(case.flash_attn_num_splits)


def _execution_name(case) -> str:
    if hasattr(case, "mode"):
        return case.mode
    return case.policy


def _is_reference_case(case) -> bool:
    return _execution_name(case) in {
        "candidate_eager_heuristic",
        "legacy_eager_auto",
    }


def _is_graph_case(case) -> bool:
    return (
        hasattr(case, "mode")
        and case.mode != "candidate_eager_heuristic"
    )


def policy_evidence(case, flash_attn_version: str) -> dict:
    return {
        "flash_attn_version": str(flash_attn_version),
        "split_policy_name": case.split_policy_name,
        "flash_attn_num_splits": case.flash_attn_num_splits,
        "comparison_policy_name": (
            "same_policy_fixed16"
            if hasattr(case, "mode")
            else "legacy_auto_vs_fixed16"
        ),
    }


def _flash_attn_version() -> str:
    try:
        import flash_attn
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}"
    return str(getattr(flash_attn, "__version__", "unknown"))


def _load_reference_tokens(path: Path, case) -> list[list[int]] | None:
    if _is_reference_case(case):
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
    if hasattr(case, "mode"):
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


def _run_with_kv_slot_restore(runner, physical_slots, operation):
    slots = sorted(set(int(slot) for slot in physical_slots))
    snapshot = runner.snapshot_kv_slots(slots)
    try:
        return operation()
    finally:
        _restore_kv_slots(runner, slots, snapshot)


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


def _run_with_split_policy(num_splits: int, operation):
    from tinyvllm.utils.context import temporary_flash_attn_num_splits

    with temporary_flash_attn_num_splits(num_splits):
        return operation()


def _allocate_decode_graph_tensors(
    runner,
    identity: split_policy.FlashAttentionGraphIdentity,
    *,
    torch_module=None,
) -> dict:
    if torch_module is None:
        import torch as torch_module

    config = runner.config
    device = runner.kv_cache.device
    graph_size = identity.graph_batch_size
    return {
        "input_ids": torch_module.zeros(
            graph_size,
            dtype=torch_module.int64,
            device=device,
        ),
        "positions": torch_module.zeros(
            graph_size,
            dtype=torch_module.int64,
            device=device,
        ),
        "slot_mapping": torch_module.zeros(
            graph_size,
            dtype=torch_module.int32,
            device=device,
        ),
        "context_lens": torch_module.zeros(
            graph_size,
            dtype=torch_module.int32,
            device=device,
        ),
        "block_tables": torch_module.zeros(
            graph_size,
            identity.page_table_width,
            dtype=torch_module.int32,
            device=device,
        ),
        "outputs": torch_module.zeros(
            graph_size,
            config.hf_config.hidden_size,
            dtype=config.hf_config.torch_dtype,
            device=device,
        ),
        "layer_outputs": torch_module.zeros(
            2,
            config.hf_config.num_hidden_layers,
            graph_size,
            config.hf_config.hidden_size,
            dtype=config.hf_config.torch_dtype,
            device=device,
        ),
    }


def _validate_graph_replay_identity(
    captured: CapturedDecodeGraph,
    identity: split_policy.FlashAttentionGraphIdentity,
) -> None:
    if captured.identity != identity:
        raise ValueError("captured graph identity mismatch")


def _get_or_capture_decode_graph(
    graph_cache: dict,
    identity: split_policy.FlashAttentionGraphIdentity,
    capture,
):
    captured = graph_cache.get(identity)
    if captured is None:
        captured = capture(identity)
        _validate_graph_replay_identity(captured, identity)
        graph_cache[identity] = captured
    else:
        _validate_graph_replay_identity(captured, identity)
    return captured


def _capture_decode_graph(
    runner,
    identity: split_policy.FlashAttentionGraphIdentity,
    dynamic_context,
):
    import torch

    from tinyvllm.utils.context import reset_context, set_context

    tensors = _allocate_decode_graph_tensors(runner, identity)
    input_ids = tensors["input_ids"]
    positions = tensors["positions"]
    slot_mapping = tensors["slot_mapping"]
    context_lens = tensors["context_lens"]
    block_tables = tensors["block_tables"]
    outputs = tensors["outputs"]
    layer_outputs = tensors["layer_outputs"]

    active_size = identity.active_batch_size
    active_columns = dynamic_context["block_tables"].size(1)
    if active_columns != identity.page_table_width:
        raise ValueError("capture page-table width does not match identity")
    input_ids[:active_size].copy_(dynamic_context["input_ids"])
    positions[:active_size].copy_(dynamic_context["positions"])
    slot_mapping[:active_size].copy_(dynamic_context["slot_mapping"])
    context_lens[:active_size].copy_(dynamic_context["context_lens"])
    block_tables[:active_size].copy_(
        dynamic_context["block_tables"]
    )

    capture_slots = sorted(
        set(int(slot) for slot in slot_mapping[:active_size].tolist())
    )
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
        def capture():
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

        _run_with_kv_slot_restore(
            runner,
            capture_slots,
            lambda: _run_with_split_policy(
                identity.effective_num_splits,
                capture,
            ),
        )
    finally:
        for handle in handles:
            handle.remove()
        reset_context()
    return CapturedDecodeGraph(
        identity=identity,
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


def _run_eager_step(runner, dynamic, num_splits: int):
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
        logits = _run_with_split_policy(
            num_splits,
            lambda: _forward_and_logits_without_autograd(
                runner.model,
                dynamic["input_ids"],
                dynamic["positions"],
            ),
        )
        torch.cuda.synchronize()
        return (
            logits.detach().cpu().contiguous(),
            layer_outputs.detach().cpu().contiguous(),
        )
    finally:
        for handle in handles:
            handle.remove()


def _run_graph_step(
    runner,
    captured,
    dynamic,
    identity: split_policy.FlashAttentionGraphIdentity,
):
    import torch

    from tinyvllm.utils.context import set_context

    _validate_graph_replay_identity(captured, identity)
    active_size = identity.active_batch_size
    active_columns = dynamic["block_tables"].size(1)
    if active_columns != identity.page_table_width:
        raise ValueError("replay page-table width does not match identity")
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
    captured.block_tables[:active_size].copy_(
        dynamic["block_tables"]
    )
    set_context(
        False,
        slot_mapping=captured.slot_mapping,
        context_lens=captured.context_lens,
        block_tables=captured.block_tables,
    )
    def replay():
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

    return _run_with_split_policy(
        identity.effective_num_splits,
        replay,
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
    evidence: dict,
) -> dict:
    shard = {
        "schema_version": 1,
        "case_id": case.case_id,
        **evidence,
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
    evidence: dict,
) -> dict:
    import torch

    shard = {
        "schema_version": 1,
        "case_id": case.case_id,
        **evidence,
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
    flash_attn_version = _flash_attn_version()
    evidence = policy_evidence(case, flash_attn_version)
    _atomic_write_json(
        output_dir / "process_environment.json",
        {
            **_process_environment(),
            **evidence,
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
        graph_cache = {}
        reset_context()

        total_steps = contract.WARMUP_STEPS + contract.MEASURED_STEPS
        raw_rows = []
        layer_rows = []
        kv_rows = []
        logits_steps = []
        layer_steps = []
        kv_step_payloads = []
        observed_reference_tokens = []
        used_step_identities = []

        for absolute_step in range(total_steps):
            if absolute_step > 0:
                _reserve_decode_slots(llm, seqs)
            dynamic = _dynamic_decode_inputs(runner, seqs)
            active_slots = [
                int(slot) for slot in dynamic["slot_mapping"].tolist()
            ]
            inactive_slots = (
                [0]
                if getattr(case, "graph_size", case.batch_size)
                > case.batch_size
                else []
            )
            observation_plan = build_kv_observation_plan(
                active_slots=active_slots,
                graph_size=getattr(case, "graph_size", case.batch_size),
                inactive_slots=inactive_slots,
                total_slots=(
                    runner.kv_cache.size(2) * runner.block_size
                ),
            )
            observation_slots = _ordered_observation_slots(
                observation_plan
            )
            before = runner.snapshot_kv_slots(observation_slots)

            if _execution_name(case) == "legacy_eager_auto":
                step_policy = None
                logits, layers = _run_eager_step(
                    runner,
                    dynamic,
                    contract.AUTO_FLASH_ATTN_NUM_SPLITS,
                )
            else:
                step_policy = build_step_split_policy(
                    runner=runner,
                    dynamic_context=dynamic,
                    active_batch_size=case.batch_size,
                    graph_batch_size=getattr(
                        case,
                        "graph_size",
                        case.batch_size,
                    ),
                    flash_attn_version=flash_attn_version,
                )
                used_step_identities.append(step_policy.identity)
                if not _is_graph_case(case):
                    logits, layers = _run_eager_step(
                        runner,
                        dynamic,
                        step_policy.effective_num_splits,
                    )
                else:
                    captured = _get_or_capture_decode_graph(
                        graph_cache,
                        step_policy.identity,
                        lambda identity: _capture_decode_graph(
                            runner,
                            identity,
                            dynamic,
                        ),
                    )
                    logits, layers = _run_graph_step(
                        runner,
                        captured,
                        dynamic,
                        step_policy.identity,
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
                case_identity = {
                    **asdict(case),
                    **evidence,
                }
                raw_row = build_step_row(
                    **case_identity,
                    case_id=case.case_id,
                    step_id=measured_step,
                    absolute_step=absolute_step,
                    execution_name=_execution_name(case),
                    active_write_slots=active_slots,
                    observed_argmax_token_ids=observed_tokens,
                    reference_next_input_token_ids=reference_row,
                )
                layer_row = {
                    **evidence,
                    "case_id": case.case_id,
                    "step_id": measured_step,
                    "required_layer_count": (
                        runner.config.hf_config.num_hidden_layers
                    ),
                    "observed_layer_count": layers.size(1),
                    "shape": list(layers.shape),
                    "finite": bool(torch.isfinite(layers).all().item()),
                }
                kv_row = {
                    **evidence,
                    "case_id": case.case_id,
                    "step_id": measured_step,
                    **observation_plan,
                    "observed_slot_ids": observation_slots,
                }
                if step_policy is not None:
                    step_rows = build_step_policy_rows(
                        step_policy,
                        raw=raw_row,
                        layer=layer_row,
                        kv=kv_row,
                    )
                    raw_row = step_rows["raw"]
                    layer_row = step_rows["layer"]
                    kv_row = step_rows["kv"]
                raw_rows.append(raw_row)
                layer_rows.append(layer_row)
                kv_rows.append(kv_row)
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

        if _is_reference_case(case):
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
            evidence=evidence,
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
            evidence=evidence,
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
            evidence=evidence,
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
            **evidence,
            "case": asdict(case),
            "case_id": case.case_id,
            "graph_identities": graph_identity_summary(
                used_step_identities
            ),
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
