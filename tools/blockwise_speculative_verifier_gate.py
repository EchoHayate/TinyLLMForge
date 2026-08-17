from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


SCHEMA_VERSION = 1
CLASSIFICATION = "NOT_PROMOTABLE"
POLICIES = ("baseline", "ngram")
CONTEXT_TOKENS = (16384, 32768)
BATCH_SIZES = (1, 4)
MAX_OUTPUT_TOKENS = 8
NGRAM_SIZE = 3
MAX_PROPOSAL_TOKENS = 4
GPU_BLOCKS = 68
LOGICAL_BLOCKS = 640
BLOCKWISE_BLOCKS = 8
REAL_MOVEMENT_KEYS = (
    "h2d_copies",
    "h2d_bytes",
    "d2h_copies",
    "d2h_bytes",
    "copy_waits",
    "evictions",
    "evict_clean",
    "speculative_residency_committed_blocks",
    "speculative_residency_rejected_blocks",
    "speculative_residency_rejected_d2h_copies",
)
DEFAULT_PROMPT_SEEDS = (
    "Blockwise verification repeats alpha beta gamma delta. ",
    "Long-context parity cycles red green blue amber. ",
    "Fixed-budget KV staging follows north east south west. ",
    "Transactional speculation echoes one two three four five. ",
)
CLAIM_SCOPE = (
    "TP1 Qwen3-0.6B 16K/32K blockwise KV-offload correctness, "
    "batch 1/4, baseline versus generic n-gram runtime"
)
LIMITATIONS = (
    "classification remains NOT_PROMOTABLE",
    "no 16K/32K performance direction",
    "no TP4 evidence",
    "no second-model evidence",
    "no learned-drafter or MTP evidence",
    "no KV8/KV4 speculative verifier evidence",
)
DEFAULT_SOURCE_FILES = (
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/speculative_residency.py",
    "tinyvllm/layers/attention.py",
    "tinyvllm/utils/context.py",
    "tools/blockwise_speculative_verifier_gate.py",
    "tools/blockwise_speculative_verifier_worker.py",
    "tools/verify_blockwise_speculative_verifier_gate.py",
)


def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a non-negative integer"
        )
    return value


def _sha256_text(value: object) -> str:
    payload = json.dumps(
        value,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def build_prompt_token_batches(
    tokenizer,
    *,
    batch_size: int,
    prompt_tokens: int,
    seeds: tuple[str, ...] = DEFAULT_PROMPT_SEEDS,
) -> list[dict]:
    batch_size = _positive_integer(batch_size, "batch_size")
    prompt_tokens = _positive_integer(
        prompt_tokens,
        "prompt_tokens",
    )
    if (
        not isinstance(seeds, tuple)
        or len(seeds) < batch_size
        or any(
            not isinstance(seed, str) or not seed
            for seed in seeds[:batch_size]
        )
    ):
        raise ValueError(
            "seed inventory must cover the requested batch"
        )
    rows = []
    for prompt_index, seed in enumerate(
        seeds[:batch_size]
    ):
        encoded = tokenizer.encode(
            seed,
            add_special_tokens=False,
        )
        if (
            not isinstance(encoded, (list, tuple))
            or not encoded
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in encoded
            )
        ):
            raise ValueError(
                f"seed {prompt_index} produced invalid token IDs"
            )
        repeats = (
            prompt_tokens + len(encoded) - 1
        ) // len(encoded)
        token_ids = (list(encoded) * repeats)[
            :prompt_tokens
        ]
        rows.append({
            "prompt_index": prompt_index,
            "seed": seed,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "sha256": _sha256_text(token_ids),
        })
    return rows


def worker_key(
    policy: str,
    context_tokens: int,
    batch_size: int,
) -> str:
    if policy not in POLICIES:
        raise ValueError("unsupported policy")
    if context_tokens not in CONTEXT_TOKENS:
        raise ValueError("unsupported context length")
    if batch_size not in BATCH_SIZES:
        raise ValueError("unsupported batch size")
    return f"{policy}:{context_tokens}:b{batch_size}"


def subtract_counter_summaries(
    before: dict,
    after: dict,
    *,
    keys: tuple[str, ...],
) -> dict[str, int]:
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("counter summaries must be mappings")
    if (
        not isinstance(keys, tuple)
        or not keys
        or any(not isinstance(key, str) or not key for key in keys)
    ):
        raise ValueError("counter keys must be a non-empty tuple")
    result = {}
    for key in keys:
        if key not in before or key not in after:
            raise ValueError(f"missing counter {key}")
        before_value = _non_negative_integer(
            before[key],
            f"{key} before",
        )
        after_value = _non_negative_integer(
            after[key],
            f"{key} after",
        )
        if after_value < before_value:
            raise ValueError(f"counter {key} decreased")
        result[key] = after_value - before_value
    return result


def _validate_prompt_rows(
    rows: object,
    *,
    batch_size: int,
    context_tokens: int,
) -> list[dict]:
    if not isinstance(rows, list) or len(rows) != batch_size:
        raise ValueError("prompt row count mismatch")
    normalized = []
    for prompt_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("prompt row must be a mapping")
        token_ids = row.get("token_ids")
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != context_tokens
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in token_ids
            )
        ):
            raise ValueError("prompt token IDs are invalid")
        if row.get("prompt_index") != prompt_index:
            raise ValueError("prompt indices are not canonical")
        if row.get("token_count") != context_tokens:
            raise ValueError("prompt token count mismatch")
        if row.get("sha256") != _sha256_text(token_ids):
            raise ValueError("prompt digest mismatch")
        normalized.append(copy.deepcopy(row))
    return normalized


def _validate_outputs(
    outputs: object,
    *,
    batch_size: int,
) -> list[list[int]]:
    if not isinstance(outputs, list) or len(outputs) != batch_size:
        raise ValueError("worker output row count mismatch")
    normalized = []
    for token_ids in outputs:
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != MAX_OUTPUT_TOKENS
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in token_ids
            )
        ):
            raise ValueError("worker output token shape mismatch")
        normalized.append(list(token_ids))
    return normalized


def validate_worker_result(worker_result: object) -> dict:
    if not isinstance(worker_result, dict):
        raise ValueError("worker result must be a mapping")
    if worker_result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("worker schema version mismatch")
    if worker_result.get("classification") != CLASSIFICATION:
        raise ValueError("worker classification mismatch")
    policy = worker_result.get("policy")
    context_tokens = worker_result.get("context_tokens")
    batch_size = worker_result.get("batch_size")
    worker_key(policy, context_tokens, batch_size)
    prompt_rows = _validate_prompt_rows(
        worker_result.get("prompt_rows"),
        batch_size=batch_size,
        context_tokens=context_tokens,
    )
    outputs = _validate_outputs(
        worker_result.get("outputs"),
        batch_size=batch_size,
    )
    runtime = worker_result.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("worker runtime must be a mapping")
    runtime_keys = (
        "proposal_rows",
        "proposed_tokens",
        "accepted_draft_tokens",
        "first_target_callbacks",
        "tail_callbacks",
    )
    normalized_runtime = {
        key: _non_negative_integer(
            runtime.get(key),
            f"runtime {key}",
        )
        for key in runtime_keys
    }
    if policy == "ngram" and any(
        normalized_runtime[key] <= 0
        for key in (
            "proposal_rows",
            "proposed_tokens",
            "accepted_draft_tokens",
            "first_target_callbacks",
            "tail_callbacks",
        )
    ):
        raise ValueError(
            "candidate runtime lacks proposal or callback evidence"
        )
    movement = worker_result.get("movement")
    if not isinstance(movement, dict):
        raise ValueError("worker movement must be a mapping")
    normalized_movement = {
        key: _non_negative_integer(
            movement.get(key),
            f"movement {key}",
        )
        for key in REAL_MOVEMENT_KEYS
    }
    visible_logical_blocks = _positive_integer(
        worker_result.get("visible_logical_blocks"),
        "visible_logical_blocks",
    )
    if (
        visible_logical_blocks > GPU_BLOCKS
        and (
            normalized_movement["h2d_copies"] <= 0
            or normalized_movement["h2d_bytes"] <= 0
        )
    ):
        raise ValueError("worker lacks real H2D movement")
    if (
        normalized_movement[
            "speculative_residency_rejected_d2h_copies"
        ]
        != 0
    ):
        raise ValueError(
            "rejected speculative blocks must not copy to host"
        )
    tokenizer_identifier = worker_result.get(
        "tokenizer_identifier"
    )
    dtype = worker_result.get("dtype")
    if (
        not isinstance(tokenizer_identifier, str)
        or not tokenizer_identifier
        or not isinstance(dtype, str)
        or not dtype
    ):
        raise ValueError("worker identity is incomplete")
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "policy": policy,
        "context_tokens": context_tokens,
        "batch_size": batch_size,
        "prompt_rows": prompt_rows,
        "outputs": outputs,
        "runtime": normalized_runtime,
        "movement": normalized_movement,
        "tokenizer_identifier": tokenizer_identifier,
        "dtype": dtype,
        "visible_logical_blocks": visible_logical_blocks,
    }


def _normalize_source_files(
    source_files: object,
) -> dict[str, str]:
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("source files must be a non-empty mapping")
    normalized = {}
    for path, digest in source_files.items():
        if not isinstance(path, str) or not path:
            raise ValueError("source path must be non-empty")
        normalized[path] = _validate_sha256(
            digest,
            f"source hash {path}",
        )
    return normalized


def build_artifact(
    *,
    worker_results: list[dict],
    environment: dict,
    source_files: dict[str, str],
) -> dict:
    if not isinstance(environment, dict):
        raise ValueError("environment must be a mapping")
    if (
        not isinstance(worker_results, list)
        or len(worker_results) != 8
    ):
        raise ValueError("artifact requires exactly eight workers")
    workers = {}
    tokenizer_identifier = None
    dtype = None
    for worker_result in worker_results:
        worker = validate_worker_result(worker_result)
        key = worker_key(
            worker["policy"],
            worker["context_tokens"],
            worker["batch_size"],
        )
        if key in workers:
            raise ValueError("duplicate worker cell")
        if tokenizer_identifier is None:
            tokenizer_identifier = worker[
                "tokenizer_identifier"
            ]
            dtype = worker["dtype"]
        elif (
            worker["tokenizer_identifier"]
            != tokenizer_identifier
            or worker["dtype"] != dtype
        ):
            raise ValueError("worker identities differ")
        workers[key] = worker
    expected_workers = {
        worker_key(policy, context_tokens, batch_size)
        for policy in POLICIES
        for context_tokens in CONTEXT_TOKENS
        for batch_size in BATCH_SIZES
    }
    if set(workers) != expected_workers:
        raise ValueError("worker inventory mismatch")

    parity = {}
    for context_tokens in CONTEXT_TOKENS:
        for batch_size in BATCH_SIZES:
            baseline = workers[
                worker_key(
                    "baseline",
                    context_tokens,
                    batch_size,
                )
            ]
            candidate = workers[
                worker_key(
                    "ngram",
                    context_tokens,
                    batch_size,
                )
            ]
            if baseline["prompt_rows"] != candidate["prompt_rows"]:
                raise ValueError("prompt parity mismatch")
            if baseline["outputs"] != candidate["outputs"]:
                raise ValueError("exact token parity mismatch")
            parity[f"{context_tokens}:b{batch_size}"] = {
                "status": "PASS",
                "output_sha256": _sha256_text(
                    baseline["outputs"]
                ),
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "classification": CLASSIFICATION,
        "claim_scope": CLAIM_SCOPE,
        "environment": {
            **copy.deepcopy(environment),
            "tokenizer_identifier": tokenizer_identifier,
            "dtype": dtype,
        },
        "campaign": {
            "tensor_parallel_size": 1,
            "context_tokens": list(CONTEXT_TOKENS),
            "batch_sizes": list(BATCH_SIZES),
            "policies": list(POLICIES),
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.0,
            "ignore_eos": True,
            "ngram_size": NGRAM_SIZE,
            "max_proposal_tokens": MAX_PROPOSAL_TOKENS,
            "gpu_blocks": GPU_BLOCKS,
            "logical_blocks": LOGICAL_BLOCKS,
            "blockwise_blocks": BLOCKWISE_BLOCKS,
        },
        "workers": workers,
        "parity": parity,
        "source_files": _normalize_source_files(
            source_files
        ),
        "limitations": list(LIMITATIONS),
    }


def validate_artifact(artifact: object) -> dict:
    if not isinstance(artifact, dict):
        raise ValueError("artifact must be a mapping")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("artifact schema version mismatch")
    if artifact.get("status") != "PASS":
        raise ValueError("artifact status must be PASS")
    if artifact.get("classification") != CLASSIFICATION:
        raise ValueError("artifact classification mismatch")
    workers = artifact.get("workers")
    if not isinstance(workers, dict):
        raise ValueError("artifact workers must be a mapping")
    rebuilt = build_artifact(
        worker_results=list(workers.values()),
        environment={
            key: copy.deepcopy(value)
            for key, value in artifact.get(
                "environment",
                {},
            ).items()
            if key not in (
                "tokenizer_identifier",
                "dtype",
            )
        },
        source_files=artifact.get("source_files"),
    )
    for key in (
        "claim_scope",
        "campaign",
        "workers",
        "parity",
        "source_files",
        "limitations",
        "environment",
    ):
        if artifact.get(key) != rebuilt[key]:
            raise ValueError(
                f"artifact derived field mismatch: {key}"
            )
    return copy.deepcopy(artifact)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def hash_source_files(
    *,
    repo_root: Path,
    source_files: tuple[str, ...],
) -> dict[str, str]:
    repo_root = Path(repo_root)
    if (
        not isinstance(source_files, tuple)
        or not source_files
    ):
        raise ValueError(
            "source_files must be a non-empty tuple"
        )
    result = {}
    for relative_path in source_files:
        if (
            not isinstance(relative_path, str)
            or not relative_path
        ):
            raise ValueError("source path must be non-empty")
        source_path = repo_root / relative_path
        if not source_path.is_file():
            raise ValueError(
                f"source file is missing: {relative_path}"
            )
        result[relative_path] = sha256_file(source_path)
    return result


def _subprocess_worker_runner(
    command,
    *,
    log_path,
    cwd,
) -> int:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(completed.returncode)


def _default_environment(
    *,
    model_path: str,
    command: list[str],
) -> dict:
    try:
        import torch

        torch_version = str(torch.__version__)
        device_name = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "unavailable"
        )
    except Exception:
        torch_version = "unavailable"
        device_name = "unavailable"
    return {
        "model_path": str(Path(model_path).resolve()),
        "model_identifier": Path(model_path).name,
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "device_name": device_name,
        "command": list(command),
    }


def run_gate(
    *,
    model_path: str,
    output_path: Path,
    repo_root: Path,
    worker_script: Path,
    worker_runner=_subprocess_worker_runner,
    python_executable: str = sys.executable,
    source_files: tuple[str, ...] = DEFAULT_SOURCE_FILES,
    environment: dict | None = None,
) -> dict:
    output_path = Path(output_path)
    repo_root = Path(repo_root)
    worker_script = Path(worker_script)
    worker_directory = output_path.parent / "workers"
    worker_directory.mkdir(parents=True, exist_ok=True)
    worker_results = []
    commands = []
    for policy in POLICIES:
        for context_tokens in CONTEXT_TOKENS:
            for batch_size in BATCH_SIZES:
                output = (
                    worker_directory
                    / (
                        f"worker-{policy}-{context_tokens}"
                        f"-b{batch_size}.json"
                    )
                )
                log_path = output.with_suffix(".log")
                command = [
                    python_executable,
                    str(worker_script),
                    "--model",
                    model_path,
                    "--policy",
                    policy,
                    "--context-tokens",
                    str(context_tokens),
                    "--batch-size",
                    str(batch_size),
                    "--out",
                    str(output),
                ]
                commands.append(command)
                status = worker_runner(
                    command,
                    log_path=log_path,
                    cwd=repo_root,
                )
                if status != 0:
                    diagnostic = {
                        "schema_version": SCHEMA_VERSION,
                        "status": "FAIL",
                        "classification": CLASSIFICATION,
                        "failure_reason": (
                            "worker_failed:"
                            f"{policy}:{context_tokens}:"
                            f"b{batch_size}"
                        ),
                        "worker_status": status,
                        "worker_log": str(log_path),
                        "commands": commands,
                        "limitations": list(LIMITATIONS),
                    }
                    atomic_write_json(output_path, diagnostic)
                    raise RuntimeError(
                        "worker failed: "
                        f"{policy}:{context_tokens}:"
                        f"b{batch_size}"
                    )
                if not output.is_file():
                    raise RuntimeError(
                        f"worker output is missing: {output}"
                    )
                try:
                    worker_results.append(json.loads(
                        output.read_text(encoding="utf-8")
                    ))
                except (OSError, json.JSONDecodeError) as error:
                    raise RuntimeError(
                        f"worker output is malformed: {output}"
                    ) from error
    command_environment = (
        _default_environment(
            model_path=model_path,
            command=[
                python_executable,
                str(Path(__file__).resolve()),
                "run",
                "--model",
                model_path,
                "--out",
                str(output_path),
            ],
        )
        if environment is None
        else copy.deepcopy(environment)
    )
    command_environment["worker_commands"] = commands
    artifact = build_artifact(
        worker_results=worker_results,
        environment=command_environment,
        source_files=hash_source_files(
            repo_root=repo_root,
            source_files=source_files,
        ),
    )
    atomic_write_json(output_path, artifact)
    return artifact


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    run_gate(
        model_path=args.model,
        output_path=Path(args.out),
        repo_root=repo_root,
        worker_script=(
            repo_root
            / "tools"
            / "blockwise_speculative_verifier_worker.py"
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
