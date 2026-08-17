"""Independent verifier for the Qwen3.5 TP1 root-logit correctness gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


RESULT_NAME = "tp1_real_root_logit_correctness.json"
REFERENCE_NAME = "reference_logits.pt"
NATIVE_NAME = "native_logits.pt"
MANIFEST_NAME = "source_manifest.json"
EXACT_INVENTORY = {
    RESULT_NAME,
    REFERENCE_NAME,
    NATIVE_NAME,
    MANIFEST_NAME,
}
MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0"
)
CONFIG_SHA256 = (
    "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4"
)
INDEX_SHA256 = (
    "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9"
)
SHARD_NAME = "model.safetensors-00001-of-00001.safetensors"
SHARD_SIZE = 4548221488
SHARD_SHA256 = (
    "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
)
MINIMUM_FREE_BYTES = 24 * 1024**3
ATOL = 2e-5
RTOL = 1e-5
TOPK = 20
PROMPTS = (
    (
        "p17",
        (
            237734, 105227, 220508, 88001, 203282, 70775, 186056,
            53549, 168830, 36323, 151604, 19097, 134378, 1871,
            117152, 232433, 99926,
        ),
        "be8a139b93467e0b0ed92999e8feec6de8fbaac4a2c4faf4786f798bb00cceb9",
    ),
    (
        "p65",
        (
            72098, 187379, 54872, 170153, 37646, 152927, 20420,
            135701, 3194, 118475, 233756, 101249, 216530, 84023,
            199304, 66797, 182078, 49571, 164852, 32345, 147626,
            15119, 130400, 245681, 113174, 228455, 95948, 211229,
            78722, 194003, 61496, 176777, 44270, 159551, 27044,
            142325, 9818, 125099, 240380, 107873, 223154, 90647,
            205928, 73421, 188702, 56195, 171476, 38969, 154250,
            21743, 137024, 4517, 119798, 235079, 102572, 217853,
            85346, 200627, 68120, 183401, 50894, 166175, 33668,
            148949, 16442,
        ),
        "2391c5bbc31e842e8c362e591458d05541b1566409f03672d192fe6a9702a264",
    ),
    (
        "synthetic",
        (
            128, 129, 255, 256, 1024, 32768, 65536, 124022,
            186033, 247787, 248043,
        ),
        "a36985347858070c7c917b110c793414192e691ffe160be66276b6022c940819",
    ),
)


class VerificationError(ValueError):
    pass


class _Checker:
    def __init__(self):
        self.count = 0

    def require(self, condition, detail):
        self.count += 1
        if not condition:
            raise VerificationError(detail)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON: {path.name}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise VerificationError(f"cannot hash artifact: {path.name}") from exc
    return digest.hexdigest()


def _token_sha256(tokens) -> str:
    payload = json.dumps(
        list(tokens),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _float32_sha256(row: torch.Tensor) -> str:
    return hashlib.sha256(
        row.detach().to(
            device="cpu",
            dtype=torch.float32,
        ).contiguous().numpy().tobytes(order="C")
    ).hexdigest()


def _load_tensor_map(path: Path, checker: _Checker, case_ids):
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise VerificationError(f"invalid tensor artifact: {path.name}") from exc
    checker.require(isinstance(payload, dict), "tensor artifact is not a mapping")
    checker.require(tuple(payload) == case_ids, "tensor case inventory mismatch")
    vocabulary = None
    for case_id, tensor in payload.items():
        checker.require(isinstance(tensor, torch.Tensor), "tensor row is invalid")
        checker.require(tensor.device.type == "cpu", "tensor row is not on CPU")
        checker.require(tensor.dtype == torch.float32, "tensor row is not FP32")
        checker.require(tensor.ndim == 1, "tensor row rank mismatch")
        checker.require(tensor.numel() >= TOPK, "tensor vocabulary is too small")
        checker.require(tensor.is_contiguous(), "tensor row is not contiguous")
        checker.require(bool(torch.isfinite(tensor).all()), "tensor row is non-finite")
        if vocabulary is None:
            vocabulary = tensor.numel()
        checker.require(
            tensor.numel() == vocabulary,
            "tensor vocabulary width mismatch",
        )
    return payload


def _comparison(native, official):
    native_ids = torch.argsort(
        native,
        descending=True,
        stable=True,
    )[:TOPK]
    official_ids = torch.argsort(
        official,
        descending=True,
        stable=True,
    )[:TOPK]
    native_id_list = [int(value) for value in native_ids.tolist()]
    official_id_list = [int(value) for value in official_ids.tolist()]
    native_values = [float(value) for value in native[native_ids].tolist()]
    official_values = [
        float(value) for value in official[official_ids].tolist()
    ]
    absolute = (native - official).abs()
    threshold = ATOL + RTOL * official.abs()
    scaled = absolute / threshold.clamp_min(
        torch.finfo(torch.float32).tiny
    )
    quantiles = torch.quantile(
        absolute,
        torch.tensor([0.5, 0.95, 0.99, 0.999], dtype=torch.float32),
    )
    cosine = torch.nn.functional.cosine_similarity(
        native.reshape(1, -1),
        official.reshape(1, -1),
    ).clamp(min=-1.0, max=1.0)
    return {
        "shape": list(native.shape),
        "source_dtype": "float32",
        "comparison_dtype": "float32",
        "native_full_logit_sha256": _float32_sha256(native),
        "official_full_logit_sha256": _float32_sha256(official),
        "native_topk_token_ids": native_id_list,
        "native_topk_logits": native_values,
        "official_topk_token_ids": official_id_list,
        "official_topk_logits": official_values,
        "native_winner_token_id": native_id_list[0],
        "native_runner_up_token_id": native_id_list[1],
        "native_winner_logit": native_values[0],
        "native_runner_up_logit": native_values[1],
        "native_winner_margin": native_values[0] - native_values[1],
        "official_winner_token_id": official_id_list[0],
        "official_runner_up_token_id": official_id_list[1],
        "official_winner_logit": official_values[0],
        "official_runner_up_logit": official_values[1],
        "official_winner_margin": official_values[0] - official_values[1],
        "max_abs_diff": float(absolute.max().item()),
        "mean_abs_diff": float(absolute.mean().item()),
        "abs_diff_percentiles": {
            name: float(value)
            for name, value in zip(
                ("p50", "p95", "p99", "p99_9"),
                quantiles.tolist(),
            )
        },
        "cosine_similarity": float(cosine.item()),
        "allclose_violation_count": int(
            (absolute > threshold).sum().item()
        ),
        "max_allclose_scaled_error": float(scaled.max().item()),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
    }


def _decision_preserved(row):
    if row["native_winner_token_id"] != row["official_winner_token_id"]:
        return False
    if (
        row["official_winner_token_id"]
        not in row["native_topk_token_ids"]
        or row["native_winner_token_id"]
        not in row["official_topk_token_ids"]
    ):
        return False
    official_margin = row["official_winner_margin"]
    native_margin = row["native_winner_margin"]
    if official_margin > 0.0:
        return native_margin > 0.0
    return (
        official_margin == 0.0
        and native_margin == 0.0
        and row["native_runner_up_token_id"]
        == row["official_runner_up_token_id"]
    )


def _verify_process(
    checker,
    row,
    *,
    worker,
    case_ids,
    vocabulary,
):
    checker.require(isinstance(row, dict), f"{worker} process row is invalid")
    checker.require(row.get("worker") == worker, f"{worker} role mismatch")
    checker.require(row.get("exit_code") == 0, f"{worker} exit code mismatch")
    checker.require(
        row.get("model_manifest_sha256") == MODEL_MANIFEST_SHA256,
        f"{worker} model identity mismatch",
    )
    checker.require(row.get("case_ids") == list(case_ids), f"{worker} case mismatch")
    checker.require(row.get("vocab_size") == vocabulary, f"{worker} vocab mismatch")
    checker.require(
        row.get("free_bytes_before", -1) >= MINIMUM_FREE_BYTES,
        f"{worker} memory preflight below 24 GiB",
    )
    checker.require(
        row.get("minimum_free_bytes") == MINIMUM_FREE_BYTES,
        f"{worker} memory floor mismatch",
    )
    checker.require(
        row.get("cleanup_complete") is True,
        f"{worker} cleanup is incomplete",
    )
    for name in (
        "pid",
        "gpu_index",
        "gpu_uuid",
        "start_timestamp",
        "finish_timestamp",
        "torch_version",
        "vmrss_kib",
        "vmhwm_kib",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        checker.require(name in row, f"{worker} process evidence missing: {name}")
    if worker == "reference":
        expected = {
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": "bfloat16",
            "attn_implementation": "eager",
            "use_cache": False,
        }
    else:
        expected = {
            "tensor_parallel_size": 1,
            "tensor_parallel_rank": 0,
            "dtype": "bfloat16",
            "recurrent_dtype": "float32",
            "engine_constructed": False,
            "model_runner_constructed": False,
            "scheduler_constructed": False,
            "sampler_constructed": False,
        }
    for name, value in expected.items():
        checker.require(
            row.get(name) == value,
            f"{worker} process contract mismatch: {name}",
        )


def verify_run(run_dir) -> dict:
    directory = Path(run_dir)
    checker = _Checker()
    checker.require(directory.is_dir(), "run directory is missing")
    actual = {
        path.name
        for path in directory.iterdir()
        if path.is_file()
    }
    checker.require(actual == EXACT_INVENTORY, "artifact inventory mismatch")

    result_path = directory / RESULT_NAME
    reference_path = directory / REFERENCE_NAME
    native_path = directory / NATIVE_NAME
    manifest_path = directory / MANIFEST_NAME
    result = _read_json(result_path)
    manifest = _read_json(manifest_path)
    checker.require(result.get("schema_version") == 1, "result schema mismatch")
    checker.require(manifest.get("schema_version") == 1, "manifest schema mismatch")

    source_hashes = manifest.get("source_file_sha256")
    checker.require(
        isinstance(source_hashes, dict) and bool(source_hashes),
        "source closure is missing",
    )
    checker.require(
        all(
            isinstance(name, str)
            and bool(name)
            and isinstance(digest, str)
            and len(digest) == 64
            for name, digest in source_hashes.items()
        ),
        "source closure is invalid",
    )
    expected_tree = hashlib.sha256(
        json.dumps(
            dict(sorted(source_hashes.items())),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    checker.require(
        manifest.get("source_tree_sha256") == expected_tree,
        "source tree identity mismatch",
    )
    for name, expected in (
        ("model_manifest_sha256", MODEL_MANIFEST_SHA256),
        ("config_sha256", CONFIG_SHA256),
        ("index_sha256", INDEX_SHA256),
        ("shard_name", SHARD_NAME),
        ("shard_size", SHARD_SIZE),
        ("shard_sha256", SHARD_SHA256),
    ):
        checker.require(manifest.get(name) == expected, f"checkpoint identity mismatch: {name}")
    artifacts = manifest.get("artifacts")
    checker.require(isinstance(artifacts, dict), "artifact hashes are missing")
    checker.require(
        set(artifacts) == {RESULT_NAME, REFERENCE_NAME, NATIVE_NAME},
        "artifact hash inventory mismatch",
    )
    for name, path in (
        (RESULT_NAME, result_path),
        (REFERENCE_NAME, reference_path),
        (NATIVE_NAME, native_path),
    ):
        entry = artifacts.get(name)
        checker.require(isinstance(entry, dict), f"artifact entry missing: {name}")
        checker.require(entry.get("size") == path.stat().st_size, f"artifact size mismatch: {name}")
        checker.require(entry.get("sha256") == _sha256(path), f"artifact SHA256 mismatch: {name}")

    expected_prompts = [
        {
            "case_id": case_id,
            "token_ids": list(tokens),
            "token_sha256": token_sha,
        }
        for case_id, tokens, token_sha in PROMPTS
    ]
    checker.require(result.get("prompts") == expected_prompts, "prompt-token drift")
    for _case_id, tokens, token_sha in PROMPTS:
        checker.require(_token_sha256(tokens) == token_sha, "frozen prompt SHA mismatch")
    checker.require(
        result.get("comparison_policy") == "bf16_decision_preserving",
        "comparison policy mismatch",
    )
    checker.require(
        result.get("tolerance") == {"atol": ATOL, "rtol": RTOL},
        "tolerance mismatch",
    )
    case_ids = tuple(case_id for case_id, _tokens, _sha in PROMPTS)
    reference = _load_tensor_map(reference_path, checker, case_ids)
    native = _load_tensor_map(native_path, checker, case_ids)
    vocabulary = reference[case_ids[0]].numel()
    checker.require(
        all(native[case_id].numel() == vocabulary for case_id in case_ids),
        "reference/native vocabulary mismatch",
    )

    processes = result.get("processes")
    checker.require(isinstance(processes, dict), "process evidence is missing")
    reference_process = processes.get("reference")
    native_process = processes.get("native")
    _verify_process(
        checker,
        reference_process,
        worker="reference",
        case_ids=case_ids,
        vocabulary=vocabulary,
    )
    _verify_process(
        checker,
        native_process,
        worker="native",
        case_ids=case_ids,
        vocabulary=vocabulary,
    )
    checker.require(
        reference_process.get("pid") != native_process.get("pid"),
        "reference and native processes must be separate",
    )

    states = result.get("state_rows")
    checker.require(
        isinstance(states, list) and len(states) == len(case_ids),
        "state evidence is missing",
    )
    for case_id, row in zip(case_ids, states):
        checker.require(isinstance(row, dict), "state row is invalid")
        checker.require(row.get("case_id") == case_id, "state case mismatch")
        for name, expected in (
            ("prepare_read_only", True),
            ("linear_layer_count", 18),
            ("changed_component_count", 36),
            ("full_attention_state_component_count", 0),
            ("commit_count", 1),
            ("release_zeroed", True),
            ("pool_binding_released", True),
        ):
            checker.require(row.get(name) == expected, f"state evidence mismatch: {name}")
    forbidden = result.get("forbidden_counters")
    checker.require(
        isinstance(forbidden, dict)
        and set(forbidden) == {
            "engine",
            "model_runner",
            "scheduler",
            "sampler",
            "generation",
        },
        "forbidden counter inventory mismatch",
    )
    for name, count in forbidden.items():
        checker.require(count == 0, f"forbidden counter is non-zero: {name}")

    stored = result.get("comparisons")
    checker.require(
        isinstance(stored, list) and len(stored) == len(case_ids),
        "comparison rows are missing",
    )
    recomputed = []
    decisions = []
    for case_id, stored_row in zip(case_ids, stored):
        row = {
            "case_id": case_id,
            **_comparison(native[case_id], reference[case_id]),
        }
        recomputed.append(row)
        checker.require(
            stored_row == row,
            f"derived metric mismatch: {case_id}",
        )
        decisions.append(_decision_preserved(row))
        checker.require(
            isinstance(row["cosine_similarity"], float)
            and math.isfinite(row["cosine_similarity"]),
            f"cosine metric invalid: {case_id}",
        )
    classification = "PASS" if all(decisions) else "NO_GO_LOGIT"
    checker.require(
        result.get("classification") == classification,
        "classification mismatch",
    )
    checker.require(
        isinstance(result.get("claim_boundary"), str)
        and "TP1" in result["claim_boundary"],
        "claim boundary is missing",
    )
    return {
        "classification": classification,
        "case_ids": list(case_ids),
        "checks": checker.count,
        "result_sha256": _sha256(result_path),
        "reference_logits_sha256": _sha256(reference_path),
        "native_logits_sha256": _sha256(native_path),
        "comparisons": recomputed,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    arguments = parser.parse_args(argv)
    result = verify_run(arguments.run_dir)
    print(f"PASS, {result['checks']} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
