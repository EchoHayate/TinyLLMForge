from __future__ import annotations

import string

import qwen35_tp4_hybrid_prefix_benchmark_v2_contract as contract


_PROFILE_SELECTION = {
    "recompute": (False, None),
    "exact_restore": (True, "exact_restore"),
    "recurrent_int8_per_row": (
        True,
        "recurrent_int8_per_row",
    ),
}


def _require_nonempty_path(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is invalid")
    return value


def _require_sha256(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in string.hexdigits for character in value)
    ):
        raise ValueError("source tree sha256 is invalid")
    return value


def build_profile_configuration(
    *,
    profile,
    case,
    workload_payload,
    authorized,
):
    try:
        hybrid_prefix_enabled, representation = (
            _PROFILE_SELECTION[profile]
        )
    except KeyError as error:
        raise ValueError(f"unsupported profile: {profile}") from error
    if profile != case.profile:
        raise ValueError("profile does not match case")
    canonical_payload = contract.workload_payload(case.workload)
    if workload_payload != canonical_payload:
        raise ValueError("workload payload does not match case")
    if authorized.get("gpu_indices") != list(
        contract.REQUIRED_GPU_INDICES
    ):
        raise ValueError("GPU indices are invalid")
    model_dir = _require_nonempty_path(
        authorized.get("model_dir"),
        "model dir",
    )
    tokenizer_dir = _require_nonempty_path(
        authorized.get("tokenizer_dir"),
        "tokenizer dir",
    )
    source_tree_sha256 = _require_sha256(
        authorized.get("source_tree_sha256")
    )
    sampling_max_tokens = canonical_payload["spec"][
        "generated_tokens"
    ]
    concurrency = 1 if case.phase == "correctness" else case.concurrency
    return {
        "model_dir": model_dir,
        "tokenizer_dir": tokenizer_dir,
        "tensor_parallel_size": 4,
        "sampling": {
            "temperature": 0.0,
            "max_tokens": sampling_max_tokens,
            "ignore_eos": True,
        },
        "prompt": canonical_payload,
        "concurrency": concurrency,
        "kv_capacity_bytes": 64 * 256,
        "repetitions": 1,
        "source_tree_sha256": source_tree_sha256,
        "gpu_indices": list(contract.REQUIRED_GPU_INDICES),
        "hybrid_prefix_enabled": hybrid_prefix_enabled,
        "representation": representation,
    }


class BenchmarkEngineAdapter:

    def __init__(
        self,
        configuration,
        authorized,
        *,
        engine_factory,
    ):
        self.configuration = configuration
        self.authorized = authorized
        self.engine = engine_factory(configuration, authorized)

    def configure_hybrid_prefix_publication_runtime(
        self,
        *,
        model_fingerprint,
        max_entries,
        max_bytes,
        timeout_s,
    ):
        if not self.configuration["hybrid_prefix_enabled"]:
            return None
        return (
            self.engine
            .configure_qwen35_hybrid_prefix_publication_runtime(
                model_fingerprint=model_fingerprint,
                max_entries=max_entries,
                max_bytes=max_bytes,
                timeout_s=timeout_s,
                representation=self.configuration["representation"],
            )
        )
