from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re


PLAN_SCHEMA = "tinyllmforge.qwen38-tp-correctness-plan.v1"
RECEIPT_SCHEMA = "tinyllmforge.qwen38-tp-correctness-runner-receipt.v1"
WORKER_RECEIPT_SCHEMA = (
    "tinyllmforge.qwen38-tp-correctness-worker-receipt.v1"
)
ROW_SCHEMA = "tinyllmforge.qwen38-tp-correctness-row.v2"
BUNDLE_SCHEMA = "tinyllmforge.qwen38-tp-correctness-bundle.v1"
CLEANUP_SCHEMA = "tinyllmforge.qwen38-tp-correctness-cleanup.v1"
SOURCE_SCHEMA = "tinyllmforge.source-manifest.v1"
MODEL_SCHEMA = "tinyllmforge.qwen38-model-manifest.v1"
DEFAULT_ATOL = 0.02
DEFAULT_RTOL = 0.01
APPROVED_REMOTE_ROOT = Path(
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
).resolve()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMAND_ORDER = (
    "official_tp1",
    "tinyllmforge_tp1",
    "tinyllmforge_tp4",
    "assemble",
    "verify",
)


def _require_below(path, root, label, *, allow_equal=False) -> Path:
    resolved = Path(path).resolve()
    root = Path(root).resolve()
    if resolved == root:
        if allow_equal:
            return resolved
        raise ValueError(f"{label} must be below approved remote root")
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} must be below approved remote root")
    return resolved


def _require_file(path, label) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{label} must be a regular file")
    return resolved


def _require_directory(path, label) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"{label} must be a directory")
    return resolved


def _require_sha256(value, label) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_integer(value, label) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _command(
    name,
    argv,
    *,
    environment,
    cwd,
    output_paths,
):
    return {
        "name": name,
        "argv": [str(value) for value in argv],
        "env": dict(environment),
        "cwd": str(cwd),
        "output_paths": [str(path) for path in output_paths],
    }


def _canonical_bytes(payload) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _atomic_write(path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}"
    )
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _prompt_sha256(prompt_token_ids) -> str:
    return hashlib.sha256(
        _canonical_bytes(list(prompt_token_ids))
    ).hexdigest()


def _topk_position(logits, *, position, topk):
    value = logits.detach().float().cpu()
    if getattr(value, "ndim", None) == 2:
        if tuple(value.shape[:1]) != (1,):
            raise ValueError("rank-0 logits batch shape mismatch")
        value = value[0]
    if getattr(value, "ndim", None) != 1:
        raise ValueError("rank-0 logits must be one-dimensional")
    if int(value.shape[0]) < topk:
        raise ValueError("rank-0 logits vocabulary is smaller than topk")
    topk_values, topk_indices = value.topk(topk)
    numeric_values = [float(item) for item in topk_values.tolist()]
    if not all(math.isfinite(item) for item in numeric_values):
        raise ValueError("rank-0 logits must be finite")
    return {
        "position": position,
        "topk_token_ids": [
            int(item) for item in topk_indices.tolist()
        ],
        "topk_logits": numeric_values,
    }


def _validate_rank_identities(rows, tensor_parallel_size):
    if not isinstance(rows, (tuple, list)):
        raise ValueError("rank identities must be a tuple or list")
    normalized = []
    required = {
        "rank",
        "gpu_index",
        "gpu_uuid",
        "expected_weight_shard_sha256",
        "loaded_weight_shard_sha256",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("rank identity schema mismatch")
        rank = row["rank"]
        gpu_index = row["gpu_index"]
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or isinstance(gpu_index, bool)
            or not isinstance(gpu_index, int)
            or gpu_index < 0
        ):
            raise ValueError("rank identity is invalid")
        if (
            not isinstance(row["gpu_uuid"], str)
            or not row["gpu_uuid"]
        ):
            raise ValueError("rank GPU UUID is invalid")
        for field in (
            "expected_weight_shard_sha256",
            "loaded_weight_shard_sha256",
        ):
            _require_sha256(row[field], field)
        normalized.append(dict(row))
    normalized.sort(key=lambda row: row["rank"])
    if [row["rank"] for row in normalized] != list(
        range(tensor_parallel_size)
    ):
        raise ValueError("rank identity inventory mismatch")
    if len({row["gpu_uuid"] for row in normalized}) != (
        tensor_parallel_size
    ):
        raise ValueError("rank GPU UUIDs must be distinct")
    if any(
        row["expected_weight_shard_sha256"]
        != row["loaded_weight_shard_sha256"]
        for row in normalized
    ):
        raise ValueError("loaded weight shard identity mismatch")
    return normalized


def _default_engine_factory(model_root, **kwargs):
    from tinyvllm.engine.llm_engine import LLMEngine

    return LLMEngine(str(model_root), **kwargs)


def _default_sampling_params_factory(**kwargs):
    from tinyvllm.sampling_params import SamplingParams

    return SamplingParams(**kwargs)


def _read_process_identity():
    return {
        "pid": os.getpid(),
        "pgid": os.getpgid(0),
    }


def _read_cuda_device_identity():
    import torch

    gpu_index = int(torch.cuda.current_device())
    gpu_uuid = getattr(
        torch.cuda.get_device_properties(gpu_index),
        "uuid",
        None,
    )
    if isinstance(gpu_uuid, bytes):
        gpu_uuid = gpu_uuid.decode("ascii")
    elif gpu_uuid is not None and not isinstance(gpu_uuid, str):
        candidate = str(gpu_uuid)
        parts = candidate.split("-")
        if (
            tuple(len(part) for part in parts) == (8, 4, 4, 4, 12)
            and all(
                character in "0123456789abcdefABCDEF"
                for part in parts
                for character in part
            )
        ):
            gpu_uuid = f"GPU-{candidate.lower()}"
    if not isinstance(gpu_uuid, str) or not gpu_uuid:
        raise RuntimeError("CUDA GPU UUID is unavailable")
    return {
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
    }


class _TransformersOfficialReference:

    def __init__(self, model_root):
        import torch
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError("official reference CUDA is unavailable")
        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_root),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        self.model = self.model.to(device=torch.device("cuda:0"))
        self.model.eval()
        self.closed = False

    def generate_step_logits(
        self,
        *,
        prompt_token_ids,
        generated_tokens,
    ):
        if self.closed:
            raise RuntimeError("official reference is closed")
        torch = self.torch
        input_ids = torch.tensor(
            [list(prompt_token_ids)],
            dtype=torch.int64,
            device=torch.device("cuda:0"),
        )
        completion = []
        step_logits = []
        past_key_values = None
        sequence_length = 0
        with torch.inference_mode():
            for _ in range(generated_tokens):
                input_length = int(input_ids.shape[-1])
                cache_position = torch.arange(
                    sequence_length,
                    sequence_length + input_length,
                    dtype=torch.int64,
                    device=torch.device("cuda:0"),
                )
                output = self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    cache_position=cache_position,
                )
                logits = getattr(output, "logits", None)
                past_key_values = getattr(
                    output,
                    "past_key_values",
                    None,
                )
                if past_key_values is None:
                    raise RuntimeError(
                        "official reference KV cache is missing"
                    )
                if (
                    getattr(logits, "ndim", None) != 3
                    or tuple(logits.shape[:2])
                    != (1, input_length)
                ):
                    raise RuntimeError(
                        "official reference logits shape mismatch"
                    )
                score = logits[:, -1, :].detach().float().cpu().clone()
                token_id = int(score.argmax(dim=-1).item())
                step_logits.append(score)
                completion.append(token_id)
                sequence_length += input_length
                input_ids = torch.tensor(
                    [[token_id]],
                    dtype=torch.int64,
                    device=torch.device("cuda:0"),
                )
        return {
            "generated_token_ids": completion,
            "step_logits": step_logits,
        }

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.model = None
        self.torch.cuda.empty_cache()


def _default_official_reference_factory(model_root):
    return _TransformersOfficialReference(model_root)


def read_engine_rank_identities(engine, *, timeout_s):
    local, acknowledgements = engine.call_model_runner_acknowledged(
        "qwen38_correctness_rank_identity",
        timeout_s=float(timeout_s),
    )
    rows = [local]
    rows.extend(
        acknowledgement.result
        for acknowledgement in acknowledgements
    )
    return _validate_rank_identities(
        rows,
        engine.model_runner.world_size,
    )


def run_tinyllmforge_worker(
    *,
    mode,
    attempt_root,
    output_path,
    process_output_path,
    model_root,
    model_repository,
    model_revision,
    source_tree_sha256,
    model_manifest_sha256,
    prompt_token_ids,
    generated_tokens,
    topk,
    tensor_parallel_size,
    timeout_s,
    engine_factory,
    sampling_params_factory,
    rank_identity_reader,
    process_identity_reader,
) -> dict:
    expected_tp = {
        "tinyllmforge_tp1": 1,
        "tinyllmforge_tp4": 4,
    }
    if mode not in expected_tp:
        raise ValueError("TinyLLMForge worker mode mismatch")
    tensor_parallel_size = _positive_integer(
        tensor_parallel_size,
        "tensor_parallel_size",
    )
    if tensor_parallel_size != expected_tp[mode]:
        raise ValueError("TinyLLMForge worker TP size mismatch")
    attempt = Path(attempt_root).resolve()
    output = _require_below(
        output_path,
        attempt,
        "attempt_root",
    )
    process_output = _require_below(
        process_output_path,
        attempt,
        "attempt_root",
    )
    model = Path(model_root).resolve()
    if not model.is_dir() or model.is_symlink():
        raise ValueError("model_root must be a directory")
    if model_repository != "Qwen/Qwen3.8-27B":
        raise ValueError("model repository mismatch")
    if not isinstance(model_revision, str) or re.fullmatch(
        r"[0-9a-f]{40}",
        model_revision,
    ) is None:
        raise ValueError("model revision must be immutable")
    source_tree_sha256 = _require_sha256(
        source_tree_sha256,
        "source_tree_sha256",
    )
    model_manifest_sha256 = _require_sha256(
        model_manifest_sha256,
        "model_manifest_sha256",
    )
    prompt_token_ids = tuple(prompt_token_ids)
    if (
        not prompt_token_ids
        or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in prompt_token_ids
        )
    ):
        raise ValueError("prompt_token_ids are invalid")
    generated_tokens = _positive_integer(
        generated_tokens,
        "generated_tokens",
    )
    topk = _positive_integer(topk, "topk")
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or float(timeout_s) <= 0.0
    ):
        raise ValueError("timeout_s must be positive")
    for dependency, label in (
        (engine_factory, "engine_factory"),
        (sampling_params_factory, "sampling_params_factory"),
        (rank_identity_reader, "rank_identity_reader"),
        (process_identity_reader, "process_identity_reader"),
    ):
        if not callable(dependency):
            raise TypeError(f"{label} must be callable")

    process_identity = process_identity_reader()
    if (
        not isinstance(process_identity, dict)
        or set(process_identity) != {"pid", "pgid"}
        or any(
            isinstance(process_identity.get(field), bool)
            or not isinstance(process_identity.get(field), int)
            or process_identity[field] <= 0
            for field in ("pid", "pgid")
        )
    ):
        raise ValueError("worker process identity is invalid")
    receipt = {
        "schema_version": WORKER_RECEIPT_SCHEMA,
        "classification": "FAIL",
        "failed_stage": "model_load",
        "failure_reason": None,
        "mode": mode,
        "tensor_parallel_size": tensor_parallel_size,
        **process_identity,
        "rank_inventory": [],
        "process_group_destroyed": False,
        "rank_exit_codes": [],
        "owned_children_remaining": [],
        "rank_cleanup_receipts": [],
    }
    engine = None
    recording_enabled = False
    error = None
    rows = None
    try:
        engine = engine_factory(
            model,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=True,
            max_num_seqs=1,
            max_model_len=len(prompt_token_ids) + generated_tokens,
        )
        if (
            getattr(getattr(engine, "model_runner", None), "rank", None)
            != 0
            or getattr(
                getattr(engine, "model_runner", None),
                "world_size",
                None,
            )
            != tensor_parallel_size
        ):
            raise ValueError("LLMEngine TP ownership mismatch")
        receipt["failed_stage"] = "rank_identity"
        identities = _validate_rank_identities(
            rank_identity_reader(
                engine,
                timeout_s=float(timeout_s),
            ),
            tensor_parallel_size,
        )
        receipt["rank_inventory"] = [
            row["rank"] for row in identities
        ]
        receipt["failed_stage"] = "decode"
        enabled = engine.enable_step_logits_authority_recording(
            True,
            timeout_s=float(timeout_s),
        )
        if enabled != {
            "enabled": True,
            "rank_inventory": list(range(tensor_parallel_size)),
        }:
            raise ValueError("step logits recording enable mismatch")
        recording_enabled = True
        sampling_params = sampling_params_factory(
            temperature=0.0,
            max_tokens=generated_tokens,
            ignore_eos=True,
        )
        engine.add_request(list(prompt_token_ids), sampling_params)
        generated_token_ids = []
        positions = []
        while not engine.is_finished():
            outputs, _ = engine.step()
            observation = getattr(
                engine,
                "last_step_observation",
                None,
            )
            sampled = (
                isinstance(observation, dict)
                and observation.get("do_sample") is True
                and any(
                    values
                    for values in observation.get(
                        "new_completion_tokens_by_seq",
                        {},
                    ).values()
                )
            )
            if sampled:
                positions.append(
                    _topk_position(
                        engine.read_step_logits_authority(),
                        position=len(positions),
                        topk=topk,
                    )
                )
            for _, token_ids in outputs:
                if len(token_ids) >= generated_tokens:
                    generated_token_ids = list(token_ids)
        if (
            len(generated_token_ids) != generated_tokens
            or len(positions) != generated_tokens
        ):
            raise ValueError(
                "TinyLLMForge generated evidence length mismatch"
            )
        if any(
            generated_token_ids[index]
            != position["topk_token_ids"][0]
            for index, position in enumerate(positions)
        ):
            raise ValueError(
                "TinyLLMForge generated token is not rank-0 argmax"
            )
        prompt_digest = _prompt_sha256(prompt_token_ids)
        rows = []
        for identity in identities:
            rank = identity["rank"]
            root = rank == 0
            rows.append({
                "schema_version": ROW_SCHEMA,
                "source_tree_sha256": source_tree_sha256,
                "model_manifest_sha256": model_manifest_sha256,
                "model_repository": model_repository,
                "model_revision": model_revision,
                "prompt_sha256": prompt_digest,
                "mode": mode,
                "dtype": "bfloat16",
                "tp_size": tensor_parallel_size,
                "rank": rank,
                "gpu_index": identity["gpu_index"],
                "gpu_uuid": identity["gpu_uuid"],
                "prompt_token_ids": list(prompt_token_ids),
                "generated_token_ids": list(generated_token_ids),
                "logits_authority": (
                    "rank0_full"
                    if root
                    else "unavailable_non_root_by_tp_design"
                ),
                "positions": positions if root else None,
                "finite_logits": True if root else None,
                "expected_weight_shard_sha256": identity[
                    "expected_weight_shard_sha256"
                ],
                "loaded_weight_shard_sha256": identity[
                    "loaded_weight_shard_sha256"
                ],
            })
    except BaseException as caught:
        error = caught
        receipt["failure_reason"] = (
            f"{type(caught).__name__}: {caught}"
        )
    finally:
        if recording_enabled:
            try:
                disabled = engine.enable_step_logits_authority_recording(
                    False,
                    timeout_s=float(timeout_s),
                )
                if disabled != {
                    "enabled": False,
                    "rank_inventory": list(
                        range(tensor_parallel_size)
                    ),
                }:
                    raise ValueError(
                        "step logits recording disable mismatch"
                    )
            except BaseException as disable_error:
                if error is None:
                    error = disable_error
                    receipt["failed_stage"] = "cleanup"
                    receipt["failure_reason"] = (
                        f"{type(disable_error).__name__}: "
                        f"{disable_error}"
                    )
        if engine is not None:
            try:
                cleanup = engine.exit()
                if not isinstance(cleanup, dict):
                    raise ValueError(
                        "LLMEngine cleanup receipt is invalid"
                    )
                for field in (
                    "process_group_destroyed",
                    "rank_exit_codes",
                    "owned_children_remaining",
                    "rank_cleanup_receipts",
                ):
                    receipt[field] = cleanup.get(field)
                cleanup_ok = (
                    receipt["process_group_destroyed"] is True
                    and receipt["rank_exit_codes"]
                    == [0] * tensor_parallel_size
                    and receipt["owned_children_remaining"] == []
                    and [
                        row.get("rank")
                        for row in receipt["rank_cleanup_receipts"]
                        if isinstance(row, dict)
                    ]
                    == list(range(tensor_parallel_size))
                    and all(
                        row.get("process_group_destroyed") is True
                        for row in receipt["rank_cleanup_receipts"]
                    )
                )
                if not cleanup_ok and error is None:
                    error = ValueError(
                        "LLMEngine cleanup receipt did not prove cleanup"
                    )
                    receipt["failed_stage"] = "cleanup"
                    receipt["failure_reason"] = (
                        f"{type(error).__name__}: {error}"
                    )
            except BaseException as cleanup_error:
                if error is None:
                    error = cleanup_error
                    receipt["failed_stage"] = "cleanup"
                    receipt["failure_reason"] = (
                        f"{type(cleanup_error).__name__}: "
                        f"{cleanup_error}"
                    )
        if error is None:
            receipt["classification"] = "PASS"
            receipt["failed_stage"] = None
            receipt["failure_reason"] = None
            _atomic_write(
                output,
                b"".join(_canonical_bytes(row) for row in rows),
            )
        _atomic_write(
            process_output,
            _canonical_bytes(receipt),
        )
    if error is not None:
        raise error
    return receipt


def run_official_worker(
    *,
    attempt_root,
    output_path,
    process_output_path,
    model_root,
    model_repository,
    model_revision,
    source_tree_sha256,
    model_manifest_sha256,
    prompt_token_ids,
    generated_tokens,
    topk,
    reference_factory,
    device_identity_reader,
    process_identity_reader,
) -> dict:
    attempt = Path(attempt_root).resolve()
    output = _require_below(output_path, attempt, "attempt_root")
    process_output = _require_below(
        process_output_path,
        attempt,
        "attempt_root",
    )
    model = Path(model_root).resolve()
    if not model.is_dir() or model.is_symlink():
        raise ValueError("model_root must be a directory")
    if model_repository != "Qwen/Qwen3.8-27B":
        raise ValueError("model repository mismatch")
    if (
        not isinstance(model_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", model_revision) is None
    ):
        raise ValueError("model revision must be immutable")
    source_tree_sha256 = _require_sha256(
        source_tree_sha256,
        "source_tree_sha256",
    )
    model_manifest_sha256 = _require_sha256(
        model_manifest_sha256,
        "model_manifest_sha256",
    )
    prompt_token_ids = tuple(prompt_token_ids)
    if (
        not prompt_token_ids
        or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in prompt_token_ids
        )
    ):
        raise ValueError("prompt_token_ids are invalid")
    generated_tokens = _positive_integer(
        generated_tokens,
        "generated_tokens",
    )
    topk = _positive_integer(topk, "topk")
    for dependency, label in (
        (reference_factory, "reference_factory"),
        (device_identity_reader, "device_identity_reader"),
        (process_identity_reader, "process_identity_reader"),
    ):
        if not callable(dependency):
            raise TypeError(f"{label} must be callable")
    process_identity = process_identity_reader()
    if (
        not isinstance(process_identity, dict)
        or set(process_identity) != {"pid", "pgid"}
        or any(
            isinstance(process_identity.get(field), bool)
            or not isinstance(process_identity.get(field), int)
            or process_identity[field] <= 0
            for field in ("pid", "pgid")
        )
    ):
        raise ValueError("worker process identity is invalid")
    receipt = {
        "schema_version": WORKER_RECEIPT_SCHEMA,
        "classification": "FAIL",
        "failed_stage": "reference",
        "failure_reason": None,
        "mode": "official_tp1",
        "tensor_parallel_size": 1,
        **process_identity,
        "rank_inventory": [],
        "process_group_destroyed": False,
        "rank_exit_codes": [],
        "owned_children_remaining": [],
        "rank_cleanup_receipts": [],
    }
    reference = None
    row = None
    error = None
    try:
        reference = reference_factory(model)
        device_identity = device_identity_reader()
        if (
            not isinstance(device_identity, dict)
            or set(device_identity) != {"gpu_index", "gpu_uuid"}
            or isinstance(device_identity["gpu_index"], bool)
            or not isinstance(device_identity["gpu_index"], int)
            or device_identity["gpu_index"] < 0
            or not isinstance(device_identity["gpu_uuid"], str)
            or not device_identity["gpu_uuid"]
        ):
            raise ValueError("official reference GPU identity is invalid")
        result = reference.generate_step_logits(
            prompt_token_ids=prompt_token_ids,
            generated_tokens=generated_tokens,
        )
        if (
            not isinstance(result, dict)
            or set(result)
            != {"generated_token_ids", "step_logits"}
        ):
            raise ValueError("official reference result schema mismatch")
        generated_token_ids = list(result["generated_token_ids"])
        logits = list(result["step_logits"])
        if (
            len(generated_token_ids) != generated_tokens
            or len(logits) != generated_tokens
        ):
            raise ValueError(
                "official reference generated evidence length mismatch"
            )
        positions = [
            _topk_position(value, position=index, topk=topk)
            for index, value in enumerate(logits)
        ]
        if any(
            generated_token_ids[index]
            != position["topk_token_ids"][0]
            for index, position in enumerate(positions)
        ):
            raise ValueError(
                "official generated token is not logits argmax"
            )
        row = {
            "schema_version": ROW_SCHEMA,
            "source_tree_sha256": source_tree_sha256,
            "model_manifest_sha256": model_manifest_sha256,
            "model_repository": model_repository,
            "model_revision": model_revision,
            "prompt_sha256": _prompt_sha256(prompt_token_ids),
            "mode": "official_tp1",
            "dtype": "bfloat16",
            "tp_size": 1,
            "rank": 0,
            "gpu_index": device_identity["gpu_index"],
            "gpu_uuid": device_identity["gpu_uuid"],
            "prompt_token_ids": list(prompt_token_ids),
            "generated_token_ids": generated_token_ids,
            "logits_authority": "rank0_full",
            "positions": positions,
            "finite_logits": True,
            "expected_weight_shard_sha256": model_manifest_sha256,
            "loaded_weight_shard_sha256": model_manifest_sha256,
        }
    except BaseException as caught:
        error = caught
        receipt["failure_reason"] = (
            f"{type(caught).__name__}: {caught}"
        )
    finally:
        if reference is not None:
            try:
                reference.close()
                receipt["process_group_destroyed"] = True
                receipt["rank_exit_codes"] = [0]
                receipt["rank_cleanup_receipts"] = [{
                    "rank": 0,
                    "process_group_destroyed": True,
                }]
            except BaseException as cleanup_error:
                if error is None:
                    error = cleanup_error
                    receipt["failed_stage"] = "cleanup"
                    receipt["failure_reason"] = (
                        f"{type(cleanup_error).__name__}: "
                        f"{cleanup_error}"
                    )
        if error is None:
            receipt["classification"] = "PASS"
            receipt["failed_stage"] = None
            receipt["failure_reason"] = None
            receipt["rank_inventory"] = [0]
            _atomic_write(output, _canonical_bytes(row))
        _atomic_write(process_output, _canonical_bytes(receipt))
    if error is not None:
        raise error
    return receipt


def _load_json_object(path, label):
    path = _require_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain an object")
    return payload


def _load_worker_rows(path, *, mode, expected_ranks):
    path = _require_file(path, f"{mode} rows")
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{mode} rows are invalid JSONL") from error
    if (
        len(rows) != len(expected_ranks)
        or any(not isinstance(row, dict) for row in rows)
        or sorted(row.get("rank") for row in rows)
        != list(expected_ranks)
        or any(row.get("mode") != mode for row in rows)
    ):
        raise ValueError(f"{mode} row inventory mismatch")
    return rows


def _validate_worker_receipt(path, *, mode, expected_ranks):
    receipt = _load_json_object(path, f"{mode} process receipt")
    expected_ranks = list(expected_ranks)
    if (
        receipt.get("schema_version") != WORKER_RECEIPT_SCHEMA
        or receipt.get("classification") != "PASS"
        or receipt.get("mode") != mode
        or receipt.get("rank_inventory") != expected_ranks
        or receipt.get("process_group_destroyed") is not True
        or receipt.get("rank_exit_codes") != [0] * len(expected_ranks)
        or receipt.get("owned_children_remaining") != []
        or [
            row.get("rank")
            for row in receipt.get("rank_cleanup_receipts", ())
            if isinstance(row, dict)
        ]
        != expected_ranks
        or any(
            row.get("process_group_destroyed") is not True
            for row in receipt.get("rank_cleanup_receipts", ())
            if isinstance(row, dict)
        )
    ):
        raise ValueError(f"{mode} process receipt did not prove cleanup")
    return receipt


def _default_bundle_verifier(root):
    from qwen38_tp_correctness import validate_correctness_bundle

    return validate_correctness_bundle(root)


def assemble_correctness_bundle(
    *,
    attempt_root,
    model_manifest_path,
    official_rows_path,
    tinyllmforge_tp1_rows_path,
    tinyllmforge_tp4_rows_path,
    official_process_path,
    tinyllmforge_tp1_process_path,
    tinyllmforge_tp4_process_path,
    source_tree_sha256,
    model_manifest_sha256,
    prompt_token_ids,
    generated_tokens,
    topk,
    atol=DEFAULT_ATOL,
    rtol=DEFAULT_RTOL,
    bundle_verifier=_default_bundle_verifier,
) -> dict:
    attempt = Path(attempt_root).resolve()
    inputs = {
        "official_tp1": (
            official_rows_path,
            official_process_path,
            (0,),
        ),
        "tinyllmforge_tp1": (
            tinyllmforge_tp1_rows_path,
            tinyllmforge_tp1_process_path,
            (0,),
        ),
        "tinyllmforge_tp4": (
            tinyllmforge_tp4_rows_path,
            tinyllmforge_tp4_process_path,
            (0, 1, 2, 3),
        ),
    }
    for mode, (row_path, process_path, _) in inputs.items():
        _require_below(row_path, attempt, f"{mode} rows")
        _require_below(
            process_path,
            attempt,
            f"{mode} process receipt",
        )
    model_manifest = _require_file(
        model_manifest_path,
        "model_manifest",
    )
    model_manifest_bytes = model_manifest.read_bytes()
    model_manifest_sha256 = _require_sha256(
        model_manifest_sha256,
        "model_manifest_sha256",
    )
    if (
        hashlib.sha256(model_manifest_bytes).hexdigest()
        != model_manifest_sha256
    ):
        raise ValueError("model manifest SHA-256 mismatch")
    try:
        model_payload = json.loads(model_manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("model manifest is invalid JSON") from error
    if (
        not isinstance(model_payload, dict)
        or model_payload.get("schema_version") != MODEL_SCHEMA
        or model_payload.get("repository") != "Qwen/Qwen3.8-27B"
        or not isinstance(model_payload.get("resolved_revision"), str)
        or re.fullmatch(
            r"[0-9a-f]{40}",
            model_payload["resolved_revision"],
        )
        is None
    ):
        raise ValueError("model manifest identity mismatch")
    source_tree_sha256 = _require_sha256(
        source_tree_sha256,
        "source_tree_sha256",
    )
    prompt_token_ids = tuple(prompt_token_ids)
    if (
        not prompt_token_ids
        or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in prompt_token_ids
        )
    ):
        raise ValueError("prompt_token_ids are invalid")
    generated_tokens = _positive_integer(
        generated_tokens,
        "generated_tokens",
    )
    topk = _positive_integer(topk, "topk")
    for value, label in ((atol, "atol"), (rtol, "rtol")):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{label} must be non-negative")
    if not callable(bundle_verifier):
        raise TypeError("bundle_verifier must be callable")

    rows = []
    cleanup_ranks = {}
    exit_codes = {}
    for mode, (row_path, process_path, expected_ranks) in inputs.items():
        mode_rows = _load_worker_rows(
            row_path,
            mode=mode,
            expected_ranks=expected_ranks,
        )
        receipt = _validate_worker_receipt(
            process_path,
            mode=mode,
            expected_ranks=expected_ranks,
        )
        rows.extend(mode_rows)
        cleanup_ranks[mode] = list(expected_ranks)
        exit_codes[mode] = list(receipt["rank_exit_codes"])

    outputs = {
        "model_manifest": attempt / "model_manifest.json",
        "source_manifest": attempt / "source_manifest.json",
        "correctness_manifest": attempt / "correctness_manifest.json",
        "correctness_rows": attempt / "correctness_rows.jsonl",
        "cleanup_receipt": attempt / "cleanup_receipt.json",
    }
    _atomic_write(outputs["model_manifest"], model_manifest_bytes)
    _atomic_write(
        outputs["source_manifest"],
        _canonical_bytes({
            "schema_version": SOURCE_SCHEMA,
            "source_tree_sha256": source_tree_sha256,
            "model_manifest_sha256": model_manifest_sha256,
        }),
    )
    _atomic_write(
        outputs["correctness_manifest"],
        _canonical_bytes({
            "schema_version": BUNDLE_SCHEMA,
            "source_tree_sha256": source_tree_sha256,
            "model_manifest_sha256": model_manifest_sha256,
            "model_repository": "Qwen/Qwen3.8-27B",
            "model_revision": model_payload["resolved_revision"],
            "prompt_token_ids": list(prompt_token_ids),
            "prompt_sha256": _prompt_sha256(prompt_token_ids),
            "dtype": "bfloat16",
            "generated_token_count": generated_tokens,
            "topk": topk,
            "atol": float(atol),
            "rtol": float(rtol),
        }),
    )
    _atomic_write(
        outputs["correctness_rows"],
        b"".join(_canonical_bytes(row) for row in rows),
    )
    _atomic_write(
        outputs["cleanup_receipt"],
        _canonical_bytes({
            "schema_version": CLEANUP_SCHEMA,
            "process_groups_destroyed": cleanup_ranks,
            "rank_exit_codes": exit_codes,
            "owned_children_remaining": [],
        }),
    )
    result = bundle_verifier(attempt)
    if (
        not isinstance(result, dict)
        or result.get("classification") != "PASS"
    ):
        raise RuntimeError(
            "assembled Qwen3.8 correctness bundle did not verify"
        )
    return result


def build_correctness_plan(
    *,
    attempt_root,
    source_root,
    model_root,
    model_manifest_path,
    source_tree_sha256,
    model_manifest_sha256,
    python_executable,
    torchrun_executable,
    gpu_indices,
    rendezvous_ports,
    prompt_token_ids,
    generated_tokens,
    topk,
    timeout_s,
) -> dict:
    approved = APPROVED_REMOTE_ROOT.resolve()
    attempt = _require_below(
        attempt_root,
        approved,
        "attempt_root",
    )
    source = _require_directory(source_root, "source_root")
    model = _require_directory(model_root, "model_root")
    manifest = _require_file(
        model_manifest_path,
        "model_manifest_path",
    )
    for path, label in (
        (source, "source_root"),
        (model, "model_root"),
        (manifest, "model_manifest_path"),
        (Path(python_executable), "python_executable"),
        (Path(torchrun_executable), "torchrun_executable"),
    ):
        _require_below(path, approved, label, allow_equal=False)

    source_tree_sha256 = _require_sha256(
        source_tree_sha256,
        "source_tree_sha256",
    )
    model_manifest_sha256 = _require_sha256(
        model_manifest_sha256,
        "model_manifest_sha256",
    )
    gpu_indices = tuple(gpu_indices)
    if (
        len(gpu_indices) != 4
        or len(set(gpu_indices)) != 4
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in gpu_indices
        )
    ):
        raise ValueError("gpu_indices must contain four distinct indices")
    rendezvous_ports = tuple(rendezvous_ports)
    if (
        len(rendezvous_ports) != 2
        or len(set(rendezvous_ports)) != 2
        or any(
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 1024 <= port <= 65535
            for port in rendezvous_ports
        )
    ):
        raise ValueError("rendezvous ports must be distinct valid ports")
    prompt_token_ids = tuple(prompt_token_ids)
    if (
        not prompt_token_ids
        or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in prompt_token_ids
        )
    ):
        raise ValueError(
            "prompt_token_ids must contain non-negative integers"
        )
    generated_tokens = _positive_integer(
        generated_tokens,
        "generated_tokens",
    )
    topk = _positive_integer(topk, "topk")
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or float(timeout_s) <= 0
    ):
        raise ValueError("timeout_s must be positive")

    python_executable = Path(python_executable).resolve()
    torchrun_executable = Path(torchrun_executable).resolve()
    runner_path = source / "tools" / "run_qwen38_tp_correctness.py"
    verifier_path = source / "tools" / "qwen38_tp_correctness.py"
    rows_dir = attempt / "rows"
    process_dir = attempt / "processes"
    model_manifest_output = attempt / "model_manifest.json"
    source_manifest_output = attempt / "source_manifest.json"
    correctness_manifest_output = attempt / "correctness_manifest.json"
    correctness_rows_output = attempt / "correctness_rows.jsonl"
    cleanup_receipt_output = attempt / "cleanup_receipt.json"
    runner_receipt_output = attempt / "runner_receipt.json"
    common = [
        f"--attempt-root={attempt}",
        f"--source-root={source}",
        f"--model-root={model}",
        f"--model-manifest={manifest}",
        f"--source-tree-sha256={source_tree_sha256}",
        f"--model-manifest-sha256={model_manifest_sha256}",
        "--text-only",
        "--greedy",
        "--temperature=0",
        f"--generated-tokens={generated_tokens}",
        "--prompt-token-ids="
        + json.dumps(list(prompt_token_ids), separators=(",", ":")),
        f"--topk={topk}",
        "--dtype=bfloat16",
        "--disable-profiler",
    ]
    official_output = rows_dir / "official_tp1.jsonl"
    tp1_output = rows_dir / "tinyllmforge_tp1.jsonl"
    tp4_output = rows_dir / "tinyllmforge_tp4.jsonl"
    official_process = process_dir / "official_tp1.json"
    tp1_process = process_dir / "tinyllmforge_tp1.json"
    tp4_process = process_dir / "tinyllmforge_tp4.json"

    commands = {
        "official_tp1": _command(
            "official_tp1",
            [
                python_executable,
                runner_path,
                "worker",
                "--mode=official_tp1",
                f"--output={official_output}",
                f"--process-output={official_process}",
                *common,
            ],
            environment={"CUDA_VISIBLE_DEVICES": str(gpu_indices[0])},
            cwd=source,
            output_paths=(official_output, official_process),
        ),
        "tinyllmforge_tp1": _command(
            "tinyllmforge_tp1",
            [
                python_executable,
                runner_path,
                "worker",
                "--mode=tinyllmforge_tp1",
                "--tensor-parallel-size=1",
                f"--dist-port={rendezvous_ports[0]}",
                f"--output={tp1_output}",
                f"--process-output={tp1_process}",
                *common,
            ],
            environment={"CUDA_VISIBLE_DEVICES": str(gpu_indices[0])},
            cwd=source,
            output_paths=(tp1_output, tp1_process),
        ),
        "tinyllmforge_tp4": _command(
            "tinyllmforge_tp4",
            [
                python_executable,
                runner_path,
                "worker",
                "--mode=tinyllmforge_tp4",
                "--tensor-parallel-size=4",
                f"--dist-port={rendezvous_ports[1]}",
                f"--output={tp4_output}",
                f"--process-output={tp4_process}",
                *common,
            ],
            environment={
                "CUDA_VISIBLE_DEVICES": ",".join(
                    str(index) for index in gpu_indices
                )
            },
            cwd=source,
            output_paths=(tp4_output, tp4_process),
        ),
        "assemble": _command(
            "assemble",
            [
                python_executable,
                runner_path,
                "assemble",
                f"--attempt-root={attempt}",
                f"--model-manifest={manifest}",
                f"--official-rows={official_output}",
                f"--tinyllmforge-tp1-rows={tp1_output}",
                f"--tinyllmforge-tp4-rows={tp4_output}",
                f"--official-process={official_process}",
                f"--tinyllmforge-tp1-process={tp1_process}",
                f"--tinyllmforge-tp4-process={tp4_process}",
                f"--source-tree-sha256={source_tree_sha256}",
                f"--model-manifest-sha256={model_manifest_sha256}",
                "--prompt-token-ids="
                + json.dumps(
                    list(prompt_token_ids),
                    separators=(",", ":"),
                ),
                f"--generated-tokens={generated_tokens}",
                f"--topk={topk}",
            ],
            environment={},
            cwd=source,
            output_paths=(
                model_manifest_output,
                source_manifest_output,
                correctness_manifest_output,
                correctness_rows_output,
                cleanup_receipt_output,
            ),
        ),
        "verify": _command(
            "verify",
            [
                python_executable,
                verifier_path,
                attempt,
            ],
            environment={},
            cwd=source,
            output_paths=(),
        ),
    }
    write_paths = [
        official_output,
        tp1_output,
        tp4_output,
        official_process,
        tp1_process,
        tp4_process,
        model_manifest_output,
        source_manifest_output,
        correctness_manifest_output,
        correctness_rows_output,
        cleanup_receipt_output,
        runner_receipt_output,
    ]
    return {
        "schema_version": PLAN_SCHEMA,
        "attempt_root": str(attempt),
        "source_root": str(source),
        "model_root": str(model),
        "model_manifest_path": str(manifest),
        "source_tree_sha256": source_tree_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "gpu_indices": list(gpu_indices),
        "rendezvous_ports": list(rendezvous_ports),
        "prompt_token_ids": list(prompt_token_ids),
        "generated_tokens": generated_tokens,
        "topk": topk,
        "timeout_s": float(timeout_s),
        "command_order": list(_COMMAND_ORDER),
        "commands": commands,
        "write_paths": [str(path) for path in write_paths],
    }


def _write_receipt(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_execution_plan(plan) -> Path:
    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("command_order") != list(_COMMAND_ORDER)
        or not isinstance(plan.get("commands"), dict)
        or set(plan["commands"]) != set(_COMMAND_ORDER)
    ):
        raise ValueError("correctness plan schema mismatch")
    attempt = _require_below(
        plan.get("attempt_root"),
        APPROVED_REMOTE_ROOT,
        "attempt_root",
    )
    source = _require_below(
        plan.get("source_root"),
        APPROVED_REMOTE_ROOT,
        "source_root",
    )
    expected_runner = source / "tools" / "run_qwen38_tp_correctness.py"
    expected_verifier = source / "tools" / "qwen38_tp_correctness.py"
    write_paths = plan.get("write_paths")
    if not isinstance(write_paths, list) or not write_paths:
        raise ValueError("write_paths inventory is invalid")
    normalized_write_paths = {
        _require_below(path, attempt, "attempt_root")
        for path in write_paths
    }
    expected_output_paths = set()
    for name in _COMMAND_ORDER:
        command = plan["commands"][name]
        if (
            not isinstance(command, dict)
            or command.get("name") != name
            or not isinstance(command.get("argv"), list)
            or not command["argv"]
            or any(
                not isinstance(argument, str) or not argument
                for argument in command["argv"]
            )
            or not isinstance(command.get("env"), dict)
            or not isinstance(command.get("output_paths"), list)
        ):
            raise ValueError(f"{name} command schema mismatch")
        executable = _require_below(
            command["argv"][0],
            APPROVED_REMOTE_ROOT,
            "command executable",
        )
        if name == "official_tp1":
            valid_entry = (
                executable.name == "python"
                and len(command["argv"]) >= 4
                and Path(command["argv"][1]).resolve()
                == expected_runner
                and command["argv"][2:4]
                == ["worker", "--mode=official_tp1"]
            )
        elif name in {"tinyllmforge_tp1", "tinyllmforge_tp4"}:
            mode = name
            valid_entry = (
                executable.name == "python"
                and len(command["argv"]) >= 4
                and Path(command["argv"][1]).resolve()
                == expected_runner
                and command["argv"][2] == "worker"
                and f"--mode={mode}" in command["argv"]
            )
        elif name == "assemble":
            valid_entry = (
                executable.name == "python"
                and len(command["argv"]) >= 3
                and Path(command["argv"][1]).resolve()
                == expected_runner
                and command["argv"][2] == "assemble"
            )
        else:
            valid_entry = (
                executable.name == "python"
                and len(command["argv"]) == 3
                and Path(command["argv"][1]).resolve()
                == expected_verifier
                and Path(command["argv"][2]).resolve() == attempt
            )
        if not valid_entry:
            raise ValueError(f"{name} command executable mismatch")
        for path in command["output_paths"]:
            expected_output_paths.add(
                _require_below(path, attempt, "attempt_root")
            )
        serialized = json.dumps(command, sort_keys=True)
        if any(
            forbidden in serialized
            for forbidden in (
                "pkill",
                "killall",
                "kinit",
                "krenew",
                "adaptive-ngram",
                "/private/tmp",
            )
        ):
            raise ValueError(f"{name} command contains a forbidden action")
    receipt_path = attempt / "runner_receipt.json"
    expected_output_paths.add(receipt_path)
    if normalized_write_paths != expected_output_paths:
        raise ValueError("write_paths inventory mismatch")
    return attempt


def _normalize_command_result(name, result):
    if not isinstance(result, dict):
        raise ValueError("command result must be a mapping")
    remaining = result.get("owned_children_remaining", [])
    if not isinstance(remaining, list):
        raise ValueError("owned_children_remaining must be a list")
    for field in ("pid", "pgid", "returncode"):
        value = result.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be an integer")
    for field in ("stdout", "stderr"):
        if not isinstance(result.get(field, ""), str):
            raise ValueError(f"{field} must be a string")
    return {
        "name": name,
        "pid": result["pid"],
        "pgid": result["pgid"],
        "returncode": result["returncode"],
        "process_group_destroyed": result.get(
            "process_group_destroyed"
        ),
        "owned_children_remaining": list(remaining),
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
    }


def execute_correctness_plan(
    plan,
    *,
    run_command,
    verify_bundle,
) -> dict:
    if not callable(run_command) or not callable(verify_bundle):
        raise ValueError("runner dependencies must be callable")
    attempt = _validate_execution_plan(plan)
    attempt.mkdir(parents=True, exist_ok=True)
    receipt_path = attempt / "runner_receipt.json"
    processes = []
    failed_stage = None
    failure_reason = None
    verification = None

    for name in _COMMAND_ORDER[:-1]:
        command = plan["commands"][name]
        try:
            result = run_command(
                command,
                timeout_s=plan["timeout_s"],
            )
            row = _normalize_command_result(name, result)
        except Exception as error:
            failed_stage = name
            failure_reason = (
                f"{type(error).__name__}: {error}"
            )
            break
        processes.append(row)
        if row["returncode"] != 0:
            failed_stage = name
            failure_reason = f"nonzero exit code: {row['returncode']}"
            break
        if row["process_group_destroyed"] is not True:
            failed_stage = name
            failure_reason = "process group cleanup was not confirmed"
            break
        if row["owned_children_remaining"]:
            failed_stage = name
            failure_reason = "owned children remain after stage"
            break

    if failed_stage is None:
        try:
            verification = verify_bundle(attempt)
        except Exception as error:
            failed_stage = "verify"
            failure_reason = (
                f"{type(error).__name__}: {error}"
            )
        else:
            if (
                not isinstance(verification, dict)
                or verification.get("classification") != "PASS"
            ):
                failed_stage = "verify"
                failure_reason = "correctness verification did not pass"

    owned_children_remaining = sorted({
        child
        for row in processes
        for child in row["owned_children_remaining"]
    })
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "classification": (
            "PASS" if failed_stage is None else "FAIL"
        ),
        "failed_stage": failed_stage,
        "failure_reason": failure_reason,
        "attempt_root": str(attempt),
        "source_tree_sha256": plan["source_tree_sha256"],
        "model_manifest_sha256": plan["model_manifest_sha256"],
        "processes": processes,
        "owned_children_remaining": owned_children_remaining,
        "verification": verification,
    }
    _write_receipt(receipt_path, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("worker", "assemble"),
        help=(
            "Execution stages are launched only by an audited controller; "
            "this module currently exposes their immutable plan contract."
        ),
    )
    parser.add_argument("--mode", dest="worker_mode")
    parser.add_argument("--attempt-root")
    parser.add_argument("--output")
    parser.add_argument("--process-output")
    parser.add_argument("--source-root")
    parser.add_argument("--model-root")
    parser.add_argument("--model-manifest")
    parser.add_argument("--official-rows")
    parser.add_argument("--tinyllmforge-tp1-rows")
    parser.add_argument("--tinyllmforge-tp4-rows")
    parser.add_argument("--official-process")
    parser.add_argument("--tinyllmforge-tp1-process")
    parser.add_argument("--tinyllmforge-tp4-process")
    parser.add_argument("--source-tree-sha256")
    parser.add_argument("--model-manifest-sha256")
    parser.add_argument("--prompt-token-ids")
    parser.add_argument("--generated-tokens", type=int)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--dist-port", type=int)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--dtype")
    parser.add_argument("--disable-profiler", action="store_true")
    args = parser.parse_args(argv)
    prompt_token_ids = json.loads(args.prompt_token_ids)
    if args.mode == "assemble":
        result = assemble_correctness_bundle(
            attempt_root=Path(args.attempt_root).resolve(),
            model_manifest_path=Path(args.model_manifest).resolve(),
            official_rows_path=Path(args.official_rows).resolve(),
            tinyllmforge_tp1_rows_path=Path(
                args.tinyllmforge_tp1_rows
            ).resolve(),
            tinyllmforge_tp4_rows_path=Path(
                args.tinyllmforge_tp4_rows
            ).resolve(),
            official_process_path=Path(
                args.official_process
            ).resolve(),
            tinyllmforge_tp1_process_path=Path(
                args.tinyllmforge_tp1_process
            ).resolve(),
            tinyllmforge_tp4_process_path=Path(
                args.tinyllmforge_tp4_process
            ).resolve(),
            source_tree_sha256=args.source_tree_sha256,
            model_manifest_sha256=args.model_manifest_sha256,
            prompt_token_ids=tuple(prompt_token_ids),
            generated_tokens=args.generated_tokens,
            topk=args.topk,
        )
        return 0 if result.get("classification") == "PASS" else 1
    if args.worker_mode not in {
        "official_tp1",
        "tinyllmforge_tp1",
        "tinyllmforge_tp4",
    }:
        raise ValueError("Qwen3.8 correctness worker mode mismatch")
    if not (
        args.text_only
        and args.greedy
        and args.temperature == 0.0
        and args.dtype == "bfloat16"
        and args.disable_profiler
    ):
        raise ValueError("Qwen3.8 worker policy mismatch")
    source_root = Path(args.source_root).resolve()
    if not source_root.is_dir() or source_root.is_symlink():
        raise ValueError("source_root must be a directory")
    manifest_path = Path(args.model_manifest).resolve()
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("model_manifest must be a regular file")
    if (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        != args.model_manifest_sha256
    ):
        raise ValueError("model manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    common = {
        "attempt_root": Path(args.attempt_root).resolve(),
        "output_path": Path(args.output).resolve(),
        "process_output_path": Path(args.process_output).resolve(),
        "model_root": Path(args.model_root).resolve(),
        "model_repository": manifest.get("repository"),
        "model_revision": manifest.get("resolved_revision"),
        "source_tree_sha256": args.source_tree_sha256,
        "model_manifest_sha256": args.model_manifest_sha256,
        "prompt_token_ids": tuple(prompt_token_ids),
        "generated_tokens": args.generated_tokens,
        "topk": args.topk,
        "process_identity_reader": _read_process_identity,
    }
    if args.worker_mode == "official_tp1":
        receipt = run_official_worker(
            **common,
            reference_factory=_default_official_reference_factory,
            device_identity_reader=_read_cuda_device_identity,
        )
        return 0 if receipt.get("classification") == "PASS" else 1
    previous_dist_port = os.environ.get("TINYVLLM_DIST_PORT")
    os.environ["TINYVLLM_DIST_PORT"] = str(args.dist_port)
    try:
        receipt = run_tinyllmforge_worker(
            mode=args.worker_mode,
            **common,
            tensor_parallel_size=args.tensor_parallel_size,
            timeout_s=args.timeout_s,
            engine_factory=_default_engine_factory,
            sampling_params_factory=_default_sampling_params_factory,
            rank_identity_reader=read_engine_rank_identities,
        )
    finally:
        if previous_dist_port is None:
            os.environ.pop("TINYVLLM_DIST_PORT", None)
        else:
            os.environ["TINYVLLM_DIST_PORT"] = previous_dist_port
    return 0 if receipt.get("classification") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
