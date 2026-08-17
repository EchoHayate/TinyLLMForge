from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys


def _load_gate_module():
    path = (
        Path(__file__).resolve().parent
        / "qwen35_generic_speculative_tp1_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "qwen35_generic_speculative_tp1_gate",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate_module()


def _integer_mapping(value: object, name: str) -> dict[int, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    normalized = {}
    for key, count in value.items():
        try:
            sequence_id = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{name} sequence ID is invalid"
            ) from error
        if (
            sequence_id < 0
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise ValueError(f"{name} count is invalid")
        normalized[sequence_id] = count
    return normalized


def normalize_side_state_receipts(
    receipts: list[dict],
) -> list[dict]:
    if not isinstance(receipts, list):
        raise ValueError(
            "side-state receipts must be a list"
        )
    normalized = []
    for receipt in receipts:
        if not isinstance(receipt, dict):
            raise ValueError(
                "side-state receipt must be a mapping"
            )
        transaction_id = receipt.get("transaction_id")
        operation = receipt.get("operation")
        status = receipt.get("status")
        sequence_ids = receipt.get("sequence_ids")
        if (
            not isinstance(transaction_id, str)
            or not transaction_id
            or operation
            not in {"prepare", "select", "apply", "seal", "rollback"}
            or not isinstance(status, str)
            or not isinstance(sequence_ids, list)
            or not sequence_ids
        ):
            raise ValueError(
                "side-state receipt is incomplete"
            )
        for sequence_id in sequence_ids:
            normalized.append({
                "sequence_id": gate._integer(
                    sequence_id,
                    "receipt sequence ID",
                ),
                "handle_id": transaction_id,
                "operation": operation,
                "state": status,
            })
    return normalized


def summarize_observations(
    observations: list[dict],
    *,
    side_state_receipts: list[dict],
) -> dict:
    summary = {
        "proposal_rows": 0,
        "proposed_tokens": 0,
        "accepted_draft_tokens": 0,
        "rejected_draft_tokens": 0,
        "first_target_callbacks": 0,
        "verify_callbacks": 0,
        "accepted_prefix_replays": 0,
        "consumed_input_mappings": [],
        "side_state_receipts": normalize_side_state_receipts(
            side_state_receipts
        ) if side_state_receipts else [],
        "failure_path_rollbacks": [],
    }
    mappings = []
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError(
                "engine observation must be a mapping"
            )
        proposal_counts = _integer_mapping(
            observation.get(
                "speculative_proposal_token_counts",
                {},
            ),
            "proposal counts",
        )
        accepted_counts = _integer_mapping(
            observation.get(
                "speculative_accepted_draft_token_counts",
                {},
            ),
            "accepted counts",
        )
        summary["proposal_rows"] += gate._integer(
            observation.get(
                "speculative_proposal_row_count",
                0,
            ),
            "proposal row count",
        )
        summary["first_target_callbacks"] += gate._integer(
            observation.get(
                "speculative_first_target_callback_count",
                0,
            ),
            "first-target callback count",
        )
        summary["verify_callbacks"] += gate._integer(
            observation.get(
                "speculative_fixed_q_group_count",
                0,
            ),
            "verify callback count",
        )
        for sequence_id, proposal_count in proposal_counts.items():
            accepted_count = accepted_counts.get(sequence_id, 0)
            if accepted_count > proposal_count:
                raise ValueError(
                    "accepted count exceeds proposal count"
                )
            verify_count = max(0, proposal_count - 1)
            committed_tail = min(
                accepted_count,
                verify_count,
            )
            mappings.append({
                "sequence_id": sequence_id,
                "proposal_token_count": proposal_count,
                "accepted_draft_count": accepted_count,
                "verify_input_count": verify_count,
                "committed_tail_input_count": committed_tail,
                "committed_input_count": 1 + committed_tail,
            })
            summary["proposed_tokens"] += proposal_count
            summary["accepted_draft_tokens"] += accepted_count
            summary["rejected_draft_tokens"] += (
                proposal_count - accepted_count
            )
    summary["consumed_input_mappings"] = mappings
    return summary


@contextmanager
def capture_side_state_receipts(model_runner):
    original_call = model_runner.call
    captured = []
    method_names = {
        "prepare_speculative_side_state_batch",
        "select_speculative_side_state_batch",
        "apply_speculative_side_state_batch",
        "seal_speculative_side_state_batch",
        "rollback_speculative_side_state_batch",
    }

    def recorded_call(method_name, *args, **kwargs):
        result = original_call(method_name, *args, **kwargs)
        if method_name in method_names:
            if not isinstance(result, dict):
                raise RuntimeError(
                    "side-state callback did not return a receipt"
                )
            captured.append(dict(result))
        return result

    model_runner.call = recorded_call
    try:
        yield captured
    finally:
        model_runner.call = original_call


def _encode_seed(tokenizer, text: str) -> list[int]:
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )
    if not isinstance(token_ids, list) or len(token_ids) < 4:
        raise RuntimeError(
            "tokenizer did not produce a usable prompt seed"
        )
    return [int(token_id) for token_id in token_ids]


def build_prompt_rows(tokenizer, batch_size: int) -> list[dict]:
    gate.cell_key("baseline", batch_size)
    target_tokens = 4048
    rows = []
    for prompt_index in range(batch_size):
        acceptance = _encode_seed(
            tokenizer,
            f" repeated alpha {prompt_index} beta gamma delta",
        )
        divergence = _encode_seed(
            tokenizer,
            f" divergent omega {prompt_index} sigma tau lambda",
        )
        pattern = acceptance + divergence
        token_ids = (
            pattern
            * ((target_tokens // len(pattern)) + 1)
        )[:target_tokens]
        token_ids[-len(acceptance):] = acceptance
        rows.append({
            "prompt_index": prompt_index,
            "token_count": len(token_ids),
            "token_ids": token_ids,
            "sha256": gate._json_sha256(token_ids),
        })
    return rows


def _model_identity(engine) -> dict:
    config = getattr(engine, "config", None)
    if config is None:
        config = getattr(engine.model_runner, "config", None)
    hf_config = getattr(config, "hf_config", None)
    if hf_config is None:
        raise RuntimeError(
            "loaded model Hugging Face config is unavailable"
        )
    text_config = getattr(
        hf_config,
        "text_config",
        hf_config,
    )
    layer_types = tuple(
        getattr(text_config, "layer_types", ())
    )
    return {
        "model_type": str(
            getattr(hf_config, "model_type", "")
        ),
        "architectures": list(
            getattr(hf_config, "architectures", ()) or ()
        ),
        "linear_layer_count": layer_types.count(
            "linear_attention"
        ),
        "full_attention_layer_count": layer_types.count(
            "full_attention"
        ),
    }


def _lease_snapshot(engine) -> dict:
    allocator = getattr(
        engine.scheduler,
        "hybrid_state_allocator",
        None,
    )
    if allocator is None:
        raise RuntimeError(
            "Qwen3.5 hybrid-state allocator is unavailable"
        )
    return allocator.observation_snapshot()


def _run_generation(
    *,
    engine,
    prompt_rows: list[dict],
    sampling_params,
    synchronize,
) -> tuple[list[dict], list[dict]]:
    for row in prompt_rows:
        engine.add_request(
            row["token_ids"],
            sampling_params,
        )
    outputs_by_id = {}
    observations = []
    while not engine.is_finished():
        step_outputs, _ = engine.step()
        synchronize()
        observation = getattr(
            engine,
            "last_step_observation",
            None,
        )
        if not isinstance(observation, dict):
            raise RuntimeError(
                "engine step observation is unavailable"
            )
        observations.append(dict(observation))
        for sequence_id, token_ids in step_outputs:
            outputs_by_id[int(sequence_id)] = [
                int(token_id)
                for token_id in token_ids
            ]
    output_rows = []
    for prompt_index, sequence_id in enumerate(
        sorted(outputs_by_id)
    ):
        token_ids = outputs_by_id[sequence_id]
        output_rows.append({
            "prompt_index": prompt_index,
            "token_count": len(token_ids),
            "token_ids": token_ids,
            "sha256": gate._json_sha256(token_ids),
        })
    if len(output_rows) != len(prompt_rows):
        raise RuntimeError(
            "engine output inventory does not match prompts"
        )
    return output_rows, observations


def run_policy_cell(
    *,
    model_path: str,
    gpu_index: int,
    policy: str,
    batch_size: int,
    engine_factory,
    sampling_params_type,
    runtime_type,
    adapter_type,
    synchronize,
) -> dict:
    gate.cell_key(policy, batch_size)
    gpu_index = gate._integer(gpu_index, "GPU index")
    engine = None
    cell = None
    cleanup = None
    try:
        engine = engine_factory(
            model_path,
            tensor_parallel_size=1,
            enforce_eager=True,
            max_model_len=4096,
            max_num_batched_tokens=8192,
            max_num_seqs=batch_size,
            kv_offload_mvp0=False,
        )
        if policy == "ngram":
            engine.activate_speculative_runtime(
                runtime_type(
                    adapter_type(
                        ngram_size=gate.NGRAM_SIZE,
                        max_proposal_tokens=(
                            gate.MAX_PROPOSAL_TOKENS
                        ),
                    )
                )
            )
        prompt_rows = build_prompt_rows(
            engine.tokenizer,
            batch_size,
        )
        sampling_params = sampling_params_type(
            temperature=0.0,
            max_tokens=gate.MAX_OUTPUT_TOKENS,
            ignore_eos=True,
        )
        before = _lease_snapshot(engine)
        with capture_side_state_receipts(
            engine.model_runner
        ) as side_state_receipts:
            output_rows, observations = _run_generation(
                engine=engine,
                prompt_rows=prompt_rows,
                sampling_params=sampling_params,
                synchronize=synchronize,
            )
        engine.flush_pending_hybrid_state_releases(
            timeout_s=60.0
        )
        after = _lease_snapshot(engine)
        runtime = summarize_observations(
            observations,
            side_state_receipts=side_state_receipts,
        )
        cell = {
            "schema_version": gate.SCHEMA_VERSION,
            "policy": policy,
            "batch_size": batch_size,
            "world_size": gate.WORLD_SIZE,
            "gpu_index": gpu_index,
            "model_identity": _model_identity(engine),
            "prompt_rows": prompt_rows,
            "output_rows": output_rows,
            "runtime": runtime,
            "lease_inventory": {
                "before": int(before["used_slots"]),
                "after": int(after["used_slots"]),
                "leaked_sequence_ids": sorted(
                    int(request_id)
                    for request_id in after["owners"].values()
                ),
            },
            "runtime_poisoned": bool(
                engine.speculative_runtime_poisoned
            ),
        }
    finally:
        if engine is not None:
            cleanup = engine.exit()
    if cell is None:
        raise RuntimeError(
            "worker did not produce a cell result"
        )
    cell["cleanup_receipt"] = {
        "engine_exit_called": cleanup is not None,
        "worker_exit_code": 0,
        "owned_children_remaining": list(
            cleanup.get("owned_children_remaining", [])
        ),
        "process_group_destroyed": bool(
            cleanup.get("process_group_destroyed", False)
        ),
        "rank_exit_codes": list(
            cleanup.get("rank_exit_codes", [])
        ),
    }
    return gate.validate_cell_result(cell)


def _default_dependencies():
    import torch

    from tinyvllm import LLM
    from tinyvllm.engine.speculative_runtime import (
        EngineSpeculativeRuntime,
    )
    from tinyvllm.sampling_params import SamplingParams
    from tinyvllm.speculative.ngram_adapter import (
        NGramDraftAdapter,
    )

    return {
        "engine_factory": LLM,
        "sampling_params_type": SamplingParams,
        "runtime_type": EngineSpeculativeRuntime,
        "adapter_type": NGramDraftAdapter,
        "synchronize": torch.cuda.synchronize,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-index", required=True, type=int)
    parser.add_argument(
        "--policy",
        required=True,
        choices=gate.POLICIES,
    )
    parser.add_argument(
        "--batch-size",
        required=True,
        type=int,
        choices=gate.BATCH_SIZES,
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_policy_cell(
        model_path=args.model,
        gpu_index=args.gpu_index,
        policy=args.policy,
        batch_size=args.batch_size,
        **_default_dependencies(),
    )
    gate.atomic_write_json(Path(args.out), result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
