from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_tools = Path(__file__).resolve().parent
gate = _load_module(
    "qwen35_generic_speculative_tp4_16k_gate",
    _tools / "qwen35_generic_speculative_tp4_16k_gate.py",
)
_frozen_worker = _load_module(
    "_qwen35_generic_speculative_tp4_frozen_worker",
    _tools / "qwen35_generic_speculative_tp4_worker.py",
)
_frozen_worker.gate = gate

for _name, _value in vars(_frozen_worker).items():
    if not _name.startswith("__") and _name != "gate":
        globals()[_name] = _value


_frozen_run_policy_cell = _frozen_worker.run_policy_cell


def run_policy_cell(
    *,
    model_path: str,
    gpu_indices: tuple[int, ...],
    policy: str,
    batch_size: int,
    dist_port: int,
    master_port: int,
    engine_factory,
    sampling_params_type,
    runtime_type,
    adapter_type,
    synchronize,
    run_generation_fn=run_generation,
) -> dict:
    def long_context_engine_factory(
        selected_model_path,
        **kwargs,
    ):
        kwargs.update({
            "max_model_len": gate.MAX_MODEL_LEN,
            "max_num_batched_tokens": (
                gate.MAX_NUM_BATCHED_TOKENS
            ),
            "max_num_prefill_tokens_per_step": (
                gate.MAX_NUM_PREFILL_TOKENS_PER_STEP
            ),
            "chunked_prefill_decode_first": False,
            "chunked_prefill_mixed_batch": False,
            "kv_offload_gpu_blocks": (
                gate.KV_OFFLOAD_GPU_BLOCKS
            ),
            "kv_offload_logical_blocks": (
                gate.KV_OFFLOAD_LOGICAL_BLOCKS
            ),
            "kv_offload_blockwise_blocks": (
                gate.KV_OFFLOAD_BLOCKWISE_BLOCKS
            ),
        })
        return engine_factory(
            selected_model_path,
            **kwargs,
        )

    return _frozen_run_policy_cell(
        model_path=model_path,
        gpu_indices=gpu_indices,
        policy=policy,
        batch_size=batch_size,
        dist_port=dist_port,
        master_port=master_port,
        engine_factory=long_context_engine_factory,
        sampling_params_type=sampling_params_type,
        runtime_type=runtime_type,
        adapter_type=adapter_type,
        synchronize=synchronize,
        run_generation_fn=run_generation_fn,
    )


_frozen_worker.run_policy_cell = run_policy_cell


if __name__ == "__main__":
    sys.exit(main())
