"""Independent verifier for the Qwen3.5 TP4 root-logit correctness gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


RESULT_NAME = "tp4_real_root_logit_correctness.json"
REFERENCE_NAME = "reference_logits.pt"
NATIVE_NAME = "native_rank0_logits.pt"
RANK_EVIDENCE_NAME = "rank_evidence.json"
MANIFEST_NAME = "source_manifest.json"
EXACT_INVENTORY = {
    RESULT_NAME,
    REFERENCE_NAME,
    NATIVE_NAME,
    RANK_EVIDENCE_NAME,
    MANIFEST_NAME,
}
SCHEMA_VERSION = "qwen35.tp4-real-root-logit-correctness.v1"
MODEL_VOCAB_SIZE = 248320
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
ATOL = 2e-5
RTOL = 0.0
COSINE_REDUCTION_ATOL = 1e-4
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
TP1_PREREQUISITE = {
    "run_tag": "qwen35-tp1-authority-20260728-195153-r2",
    "classification": "PASS",
    "source_tree_sha256": (
        "e5da50970951b32a61ecc9f85cf1cd447dc95950856345d780cd9e089bdf11ab"
    ),
    "artifacts": {
        "tp1_real_root_logit_correctness.json": (
            "39c4bbc548a82e915609dccb57101a6e64cbc92319d5ba69c44d427f8f4aa519"
        ),
        "reference_logits.pt": (
            "3373ab6038d6f21cf6421aa0e1c9146cc001e6e8bcbd06e6ab59c65ecc709e5a"
        ),
        "native_logits.pt": (
            "5d3fcb3a204a1b1bd673026d83f06ddc76422ca49d51d6af32671ba153ecb4d4"
        ),
        "source_manifest.json": (
            "0633a6ad5913d0d8a28526c1ec05f2cb17e347c180a6c93fa58fc3674fcb2207"
        ),
    },
}


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
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid JSON: {path.name}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise VerificationError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _token_sha256(tokens) -> str:
    return hashlib.sha256(
        json.dumps(
            list(tokens),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


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
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise VerificationError(
            f"invalid tensor artifact: {path.name}"
        ) from error
    checker.require(isinstance(payload, dict), "tensor artifact is not a mapping")
    checker.require(tuple(payload) == case_ids, "tensor case inventory mismatch")
    for case_id in case_ids:
        tensor = payload.get(case_id)
        checker.require(isinstance(tensor, torch.Tensor), "tensor row is invalid")
        checker.require(tensor.device.type == "cpu", "tensor row is not on CPU")
        checker.require(tensor.dtype == torch.float32, "tensor row is not FP32")
        checker.require(tensor.ndim == 1, "tensor row rank mismatch")
        checker.require(
            tensor.numel() == MODEL_VOCAB_SIZE,
            "tensor vocabulary width mismatch",
        )
        checker.require(tensor.is_contiguous(), "tensor row is not contiguous")
        checker.require(bool(torch.isfinite(tensor).all()), "tensor row is non-finite")
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


def _decision_preserved(row) -> bool:
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


def _comparison_metrics_match(recorded, recomputed) -> bool:
    if (
        not isinstance(recorded, dict)
        or set(recorded) != set(recomputed)
    ):
        return False
    recorded_cosine = recorded.get("cosine_similarity")
    recomputed_cosine = recomputed["cosine_similarity"]
    if (
        isinstance(recorded_cosine, bool)
        or not isinstance(recorded_cosine, (int, float))
        or not math.isfinite(recorded_cosine)
        or not -1.0 <= recorded_cosine <= 1.0
        or abs(recorded_cosine - recomputed_cosine)
        > COSINE_REDUCTION_ATOL
    ):
        return False
    return all(
        recorded[name] == value
        for name, value in recomputed.items()
        if name != "cosine_similarity"
    )


def _verify_rank_rows(checker, rows, case_ids):
    checker.require(isinstance(rows, list), "rank evidence is not a list")
    checker.require(len(rows) == 4, "rank evidence count mismatch")
    checker.require(
        [row.get("rank") for row in rows] == [0, 1, 2, 3],
        "rank inventory mismatch",
    )
    checker.require(
        len({row.get("pid") for row in rows}) == 4,
        "rank PID uniqueness mismatch",
    )
    checker.require(
        len({row.get("gpu_index") for row in rows}) == 4,
        "rank GPU index uniqueness mismatch",
    )
    checker.require(
        len({row.get("gpu_uuid") for row in rows}) == 4,
        "rank GPU UUID uniqueness mismatch",
    )
    checker.require(
        len({row.get("process_group_nonce") for row in rows}) == 1,
        "rank process-group nonce mismatch",
    )
    checker.require(
        len({row.get("rendezvous") for row in rows}) == 1,
        "rank rendezvous mismatch",
    )
    for rank, row in enumerate(rows):
        checker.require(isinstance(row, dict), "rank row is invalid")
        checker.require(row.get("rank") == rank, "rank identity mismatch")
        checker.require(row.get("world_size") == 4, "rank world size mismatch")
        checker.require(
            type(row.get("pid")) is int and row["pid"] > 0,
            "rank PID is invalid",
        )
        checker.require(row.get("exit_code") == 0, "rank exit code mismatch")
        checker.require(
            type(row.get("gpu_index")) is int and row["gpu_index"] >= 0,
            "rank GPU index is invalid",
        )
        checker.require(
            isinstance(row.get("gpu_uuid"), str)
            and row["gpu_uuid"].startswith("GPU-"),
            "rank GPU UUID is invalid",
        )
        checker.require(
            tuple(row.get("case_ids", ())) == case_ids,
            "rank case inventory mismatch",
        )
        checker.require(
            row.get("case_barrier_count") == len(case_ids),
            "rank case barrier mismatch",
        )
        checker.require(
            row.get("final_barrier_completed") is True,
            "rank final barrier is incomplete",
        )
        checker.require(
            row.get("process_group_destroyed") is True,
            "rank process group was not destroyed",
        )
        checker.require(
            row.get("candidate_reference_dropped") is True,
            "rank candidate reference was not dropped",
        )
        checker.require(
            row.get("model_reference_dropped") is True,
            "rank model reference was not dropped",
        )
        checker.require(
            row.get("cuda_synchronized") is True,
            "rank CUDA synchronization was incomplete",
        )
        checker.require(
            row.get("cuda_cache_emptied") is True,
            "rank CUDA cache was not emptied",
        )
        topology = {
            "global_query_heads": 8,
            "global_kv_heads": 2,
            "local_query_heads": 2,
            "local_kv_heads": 1,
            "kv_head_replicas": 2,
            "source_kv_rank": rank // 2,
        }
        for name, value in topology.items():
            checker.require(
                row.get(name) == value,
                f"rank topology mismatch: {name}",
            )
        if rank == 0:
            checker.require(
                row.get("root_logits_present") is True,
                "rank zero logits claim is invalid",
            )
            checker.require(
                row.get("non_root_logits_none") is False,
                "rank zero non-root claim is invalid",
            )
        else:
            checker.require(
                row.get("root_logits_present") is False,
                "non-root tensor-output claim is invalid",
            )
            checker.require(
                row.get("non_root_logits_none") is True,
                "non-root logits None claim is invalid",
            )
        events = row.get("collective_events")
        checker.require(
            isinstance(events, list) and bool(events),
            "rank collective evidence is missing",
        )
        checker.require(
            [event.get("ordinal") for event in events]
            == list(range(len(events))),
            "rank collective order is invalid",
        )
        checker.require(
            any(event.get("collective") == "all_reduce" for event in events),
            "rank all_reduce collective evidence is missing",
        )
        gathers = [
            event for event in events
            if event.get("collective") == "gather"
        ]
        checker.require(
            len(gathers) == 1,
            "rank gather collective count mismatch",
        )
        for event in events:
            checker.require(
                event.get("collective") in ("all_reduce", "gather"),
                "rank collective type is invalid",
            )
            checker.require(
                isinstance(event.get("shape"), list)
                and bool(event["shape"]),
                "rank collective shape is invalid",
            )
            checker.require(
                event.get("async_op") is False,
                "rank collective async behavior changed",
            )
        for event in gathers:
            checker.require(
                event.get("destination") == 0,
                "rank gather destination is invalid",
            )
            checker.require(
                event.get("receive_count") == (4 if rank == 0 else None),
                "rank gather receive count is invalid",
            )
            checker.require(
                event.get("shape") == [62080, 2048]
                and event.get("dtype") == "torch.bfloat16",
                "rank exact lm_head gather evidence is invalid",
            )
        state_rows = row.get("state_rows")
        checker.require(
            isinstance(state_rows, list)
            and len(state_rows) == len(case_ids),
            "rank state evidence count mismatch",
        )
        for case_id, state in zip(case_ids, state_rows):
            checker.require(
                state.get("case_id") == case_id,
                "rank state case identity mismatch",
            )
            checker.require(
                state.get("changed_component_count") == 36,
                "rank state component count is invalid",
            )
            values = state.get("state_nonzero_after_commit")
            checker.require(
                isinstance(values, dict)
                and len(values) == 36
                and all(value is True for value in values.values()),
                "rank state mutation evidence is invalid",
            )
            convolution_layers = {
                int(key.split(":", 1)[0])
                for key in values
                if key.endswith(":linear_convolution")
            }
            recurrent_layers = {
                int(key.split(":", 1)[0])
                for key in values
                if key.endswith(":linear_recurrent")
            }
            checker.require(
                convolution_layers == set(range(18))
                and recurrent_layers == set(range(18)),
                "rank state layer evidence is invalid",
            )
            checker.require(
                state.get("release_zeroed") is True,
                "rank state release zeroing failed",
            )
            checker.require(
                state.get("pool_binding_released") is True,
                "rank state pool binding survived",
            )


def verify_run(run_dir, *, source_root=None):
    checker = _Checker()
    directory = Path(run_dir)
    checker.require(directory.is_dir(), "run directory is missing")
    inventory = {path.name for path in directory.iterdir()}
    checker.require(
        inventory == EXACT_INVENTORY,
        "artifact inventory mismatch",
    )
    result = _read_json(directory / RESULT_NAME)
    ranks = _read_json(directory / RANK_EVIDENCE_NAME)
    manifest = _read_json(directory / MANIFEST_NAME)
    checker.require(isinstance(result, dict), "result document is invalid")
    checker.require(
        result.get("schema_version") == SCHEMA_VERSION,
        "result schema mismatch",
    )
    checker.require(
        result.get("classification") == "PASS",
        "result classification is not PASS",
    )
    checker.require(
        result.get("comparison_policy") == "registered_logits_strict_allclose",
        "comparison policy mismatch",
    )
    checker.require(
        result.get("tolerance") == {"atol": ATOL, "rtol": RTOL},
        "comparison tolerance mismatch",
    )
    expected_prompts = [
        {
            "case_id": case_id,
            "token_ids": list(tokens),
            "token_sha256": digest,
        }
        for case_id, tokens, digest in PROMPTS
    ]
    checker.require(
        result.get("prompts") == expected_prompts,
        "prompt contract mismatch",
    )
    for case_id, tokens, digest in PROMPTS:
        checker.require(
            _token_sha256(tokens) == digest,
            f"prompt token hash mismatch: {case_id}",
        )
    forbidden = {
        "engine": 0,
        "generation": 0,
        "model_runner": 0,
        "sampler": 0,
        "scheduler": 0,
    }
    checker.require(
        result.get("forbidden_counters") == forbidden,
        "forbidden counters mismatch",
    )
    reference_process = result.get("reference_process")
    checker.require(
        isinstance(reference_process, dict),
        "reference process row is invalid",
    )
    expected_reference = {
        "worker": "reference",
        "exit_code": 0,
        "model_manifest_sha256": MODEL_MANIFEST_SHA256,
        "case_ids": [case_id for case_id, _tokens, _digest in PROMPTS],
        "vocab_size": MODEL_VOCAB_SIZE,
        "cleanup_complete": True,
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "use_cache": False,
    }
    for name, value in expected_reference.items():
        checker.require(
            reference_process.get(name) == value,
            f"reference process mismatch: {name}",
        )
    checker.require(
        type(reference_process.get("pid")) is int
        and reference_process["pid"] > 0,
        "reference process PID is invalid",
    )
    checker.require(
        type(reference_process.get("gpu_index")) is int
        and reference_process["gpu_index"] >= 0,
        "reference process GPU index is invalid",
    )
    checker.require(
        isinstance(reference_process.get("gpu_uuid"), str)
        and reference_process["gpu_uuid"].startswith("GPU-"),
        "reference process GPU UUID is invalid",
    )
    checker.require(
        isinstance(result.get("claim_boundary"), str)
        and "no cached decode" in result["claim_boundary"],
        "claim boundary mismatch",
    )
    case_ids = tuple(case_id for case_id, _tokens, _digest in PROMPTS)
    reference = _load_tensor_map(
        directory / REFERENCE_NAME,
        checker,
        case_ids,
    )
    native = _load_tensor_map(
        directory / NATIVE_NAME,
        checker,
        case_ids,
    )
    comparisons = [
        {
            "case_id": case_id,
            **_comparison(native[case_id], reference[case_id]),
        }
        for case_id in case_ids
    ]
    checker.require(
        isinstance(result.get("comparisons"), list)
        and len(result["comparisons"]) == len(comparisons)
        and all(
            _comparison_metrics_match(recorded, recomputed)
            for recorded, recomputed in zip(
                result["comparisons"],
                comparisons,
            )
        ),
        "comparison metrics mismatch",
    )
    checker.require(
        all(
            _decision_preserved(row)
            and row["allclose_violation_count"] == 0
            for row in comparisons
        ),
        "comparison is not strict-allclose",
    )
    _verify_rank_rows(checker, ranks, case_ids)
    checker.require(
        reference_process["pid"] not in {row["pid"] for row in ranks},
        "reference and native PID uniqueness mismatch",
    )

    checker.require(
        isinstance(manifest, dict)
        and manifest.get("schema_version") == 1,
        "source manifest schema mismatch",
    )
    expected_checkpoint = {
        "model_manifest_sha256": MODEL_MANIFEST_SHA256,
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
        "shard_name": SHARD_NAME,
        "shard_size": SHARD_SIZE,
        "shard_sha256": SHARD_SHA256,
    }
    for name, value in expected_checkpoint.items():
        checker.require(
            manifest.get(name) == value,
            f"checkpoint identity mismatch: {name}",
        )
    checker.require(
        manifest.get("prerequisites")
        == {"tp1_real_root_logit_correctness": TP1_PREREQUISITE},
        "prerequisite identity mismatch",
    )
    artifacts = manifest.get("artifacts")
    checker.require(
        isinstance(artifacts, dict)
        and set(artifacts) == EXACT_INVENTORY - {MANIFEST_NAME},
        "artifact manifest inventory mismatch",
    )
    for name, row in artifacts.items():
        path = directory / name
        checker.require(isinstance(row, dict), "artifact manifest row is invalid")
        checker.require(
            row.get("size") == path.stat().st_size,
            f"artifact size mismatch: {name}",
        )
        checker.require(
            row.get("sha256") == _sha256(path),
            f"artifact hash mismatch: {name}",
        )
    source_hashes = manifest.get("source_file_sha256")
    checker.require(
        isinstance(source_hashes, dict) and bool(source_hashes),
        "source closure is invalid",
    )
    sorted_hashes = dict(sorted(source_hashes.items()))
    checker.require(
        manifest.get("source_tree_sha256")
        == hashlib.sha256(_canonical(sorted_hashes)).hexdigest(),
        "source tree hash mismatch",
    )
    if source_root is not None:
        root = Path(source_root)
        checker.require(root.is_dir(), "source root is missing")
        for relative, digest in sorted_hashes.items():
            source_path = root / relative
            checker.require(source_path.is_file(), "source file is missing")
            checker.require(
                _sha256(source_path) == digest,
                f"source hash mismatch: {relative}",
            )
    return {
        "classification": "PASS",
        "case_ids": list(case_ids),
        "ranks": [0, 1, 2, 3],
        "checks": checker.count,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--source-root")
    arguments = parser.parse_args(argv)
    result = verify_run(
        arguments.run_dir,
        source_root=arguments.source_root,
    )
    print(f"PASS, {result['checks']} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
