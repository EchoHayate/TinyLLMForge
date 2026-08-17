from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = (
    ROOT
    / "tools/qwen35_tp4_real_root_logit_correctness_preflight.py"
)
VERIFIER_PATH = (
    ROOT
    / "tools/verify_qwen35_tp4_real_root_logit_correctness_gate.py"
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.fspath(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preflight = _load("qwen35_tp4_verifier_fixture_preflight", PREFLIGHT_PATH)
verifier = _load("qwen35_tp4_independent_verifier_under_test", VERIFIER_PATH)


def test_verifier_freezes_strict_registered_logit_tolerance():
    assert verifier.ATOL == 2e-5
    assert verifier.RTOL == 0.0
    assert verifier.COSINE_REDUCTION_ATOL == 1e-4


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _source_manifest(source_root):
    sources = {
        "gate/contract.py": b"frozen contract\n",
        "gate/preflight.py": b"producer\n",
        "gate/verifier.py": b"independent verifier\n",
    }
    hashes = {}
    for relative, payload in sources.items():
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        hashes[relative] = hashlib.sha256(payload).hexdigest()
    return {
        "schema_version": 1,
        "source_file_sha256": hashes,
        "source_tree_sha256": hashlib.sha256(
            _canonical(dict(sorted(hashes.items())))
        ).hexdigest(),
        "model_manifest_sha256": (
            preflight.APPROVED_MODEL_MANIFEST_SHA256
        ),
        "config_sha256": preflight.APPROVED_CONFIG_SHA256,
        "index_sha256": preflight.APPROVED_INDEX_SHA256,
        "shard_name": preflight.APPROVED_SHARD_NAME,
        "shard_size": preflight.APPROVED_SHARD_SIZE,
        "shard_sha256": preflight.APPROVED_SHARD_SHA256,
        "prerequisites": {
            "tp1_real_root_logit_correctness": {
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
            },
        },
    }


def _tensor_maps():
    reference = {}
    native = {}
    for index, case in enumerate(preflight._TP4_CONTRACT.prompt_cases()):
        row = torch.linspace(
            -1.0,
            1.0,
            preflight.MODEL_VOCAB_SIZE,
            dtype=torch.float32,
        )
        row[-1] = 20.0 + index
        row[-2] = 10.0 + index
        reference[case.case_id] = row.contiguous()
        native[case.case_id] = row.clone().contiguous()
    return reference, native


def _rank_rows():
    case_ids = [
        case.case_id for case in preflight._TP4_CONTRACT.prompt_cases()
    ]
    rows = []
    for rank in range(4):
        rows.append({
            "rank": rank,
            "world_size": 4,
            "pid": 900 + rank,
            "exit_code": 0,
            "gpu_index": 40 + rank,
            "gpu_uuid": f"GPU-verifier-{rank}",
            "process_group_nonce": "v" * 32,
            "rendezvous": "tcp://127.0.0.1:45001",
            "case_ids": case_ids,
            "case_barrier_count": 3,
            "final_barrier_completed": True,
            "process_group_destroyed": True,
            "candidate_reference_dropped": True,
            "model_reference_dropped": True,
            "cuda_synchronized": True,
            "cuda_cache_emptied": True,
            "root_logits_present": rank == 0,
            "non_root_logits_none": rank != 0,
            "global_query_heads": 8,
            "global_kv_heads": 2,
            "local_query_heads": 2,
            "local_kv_heads": 1,
            "kv_head_replicas": 2,
            "source_kv_rank": rank // 2,
            "collective_events": [
                {
                    **event,
                    "ordinal": ordinal,
                }
                for ordinal, event in enumerate((
                    {
                        "collective": "all_reduce",
                        "shape": [17, 2048],
                        "dtype": "torch.bfloat16",
                        "async_op": False,
                    },
                    {
                        "collective": "gather",
                        "shape": [62080, 2048],
                        "dtype": "torch.bfloat16",
                        "destination": 0,
                        "receive_count": 4 if rank == 0 else None,
                        "async_op": False,
                    },
                    {
                        "collective": "all_reduce",
                        "shape": [65, 2048],
                        "dtype": "torch.bfloat16",
                        "async_op": False,
                    },
                    {
                        "collective": "all_reduce",
                        "shape": [9, 2048],
                        "dtype": "torch.bfloat16",
                        "async_op": False,
                    },
                ))
            ],
            "state_rows": [
                {
                    "case_id": case_id,
                    "changed_component_count": 36,
                    "state_nonzero_after_commit": {
                        **{
                            f"{layer}:linear_convolution": True
                            for layer in range(18)
                        },
                        **{
                            f"{layer}:linear_recurrent": True
                            for layer in range(18)
                        },
                    },
                    "release_zeroed": True,
                    "pool_binding_released": True,
                }
                for case_id in case_ids
            ],
        })
    return rows


def _build_run(root):
    source_root = root / "source"
    run_dir = root / "run"
    reference, native = _tensor_maps()
    preflight.finalize_tp4_correctness_artifact(
        run_dir=run_dir,
        run_tag="qwen35-tp4-verifier-test",
        reference_logits=reference,
        native_rank0_logits=native,
        reference_process={
            "worker": "reference",
            "pid": 850,
            "exit_code": 0,
            "gpu_index": 40,
            "gpu_uuid": "GPU-reference",
            "case_ids": [
                case.case_id
                for case in preflight._TP4_CONTRACT.prompt_cases()
            ],
            "vocab_size": preflight.MODEL_VOCAB_SIZE,
            "cleanup_complete": True,
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": "bfloat16",
            "attn_implementation": "eager",
            "use_cache": False,
        },
        rank_rows=_rank_rows(),
        source_manifest=_source_manifest(source_root),
        forbidden_counters={
            "engine": 0,
            "model_runner": 0,
            "scheduler": 0,
            "sampler": 0,
            "generation": 0,
        },
    )
    return run_dir, source_root


def _resign_artifact(run_dir, artifact_name):
    manifest_path = run_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifact = run_dir / artifact_name
    manifest["artifacts"][artifact_name] = {
        "size": artifact.stat().st_size,
        "sha256": _sha256(artifact),
    }
    manifest_path.write_bytes(_canonical(manifest) + b"\n")


def _expect_error(function, fragment):
    try:
        function()
    except verifier.VerificationError as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {fragment!r}")


def test_verifier_accepts_exact_five_independent_artifact():
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir, source_root = _build_run(Path(temporary_directory))
        result = verifier.verify_run(run_dir, source_root=source_root)
        assert result["classification"] == "PASS"
        assert result["case_ids"] == ["p17", "p65", "synthetic"]
        assert result["ranks"] == [0, 1, 2, 3]
        assert result["checks"] >= 100


def test_verifier_accepts_bounded_cross_backend_cosine_reduction_drift():
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir, source_root = _build_run(Path(temporary_directory))
        result_path = run_dir / "tp4_real_root_logit_correctness.json"
        result = json.loads(result_path.read_text())
        result["comparisons"][0]["cosine_similarity"] -= 3.6e-5
        result_path.write_bytes(_canonical(result) + b"\n")
        _resign_artifact(
            run_dir,
            "tp4_real_root_logit_correctness.json",
        )
        verification = verifier.verify_run(
            run_dir,
            source_root=source_root,
        )
        assert verification["classification"] == "PASS"


def test_verifier_rejects_rank_collective_state_and_cleanup_tamper():
    mutations = (
        ("rank", lambda rows: rows[3].__setitem__("rank", 2), "rank"),
        (
            "gpu",
            lambda rows: rows[3].__setitem__(
                "gpu_uuid",
                rows[2]["gpu_uuid"],
            ),
            "GPU UUID",
        ),
        (
            "nonce",
            lambda rows: rows[1].__setitem__(
                "process_group_nonce",
                "x" * 32,
            ),
            "nonce",
        ),
        (
            "world",
            lambda rows: rows[1].__setitem__("world_size", 3),
            "world size",
        ),
        (
            "topology",
            lambda rows: rows[1].__setitem__("local_query_heads", 1),
            "topology",
        ),
        (
            "non-root",
            lambda rows: rows[2].__setitem__(
                "non_root_logits_none",
                False,
            ),
            "non-root",
        ),
        (
            "collective",
            lambda rows: rows[0].__setitem__("collective_events", []),
            "collective",
        ),
        (
            "state",
            lambda rows: rows[0]["state_rows"][0].__setitem__(
                "release_zeroed",
                False,
            ),
            "state",
        ),
        (
            "cleanup",
            lambda rows: rows[0].__setitem__(
                "process_group_destroyed",
                False,
            ),
            "destroyed",
        ),
        (
            "candidate cleanup",
            lambda rows: rows[0].__setitem__(
                "candidate_reference_dropped",
                False,
            ),
            "candidate reference",
        ),
        (
            "model cleanup",
            lambda rows: rows[0].__setitem__(
                "model_reference_dropped",
                False,
            ),
            "model reference",
        ),
        (
            "CUDA synchronization",
            lambda rows: rows[0].__setitem__(
                "cuda_synchronized",
                False,
            ),
            "CUDA synchronization",
        ),
        (
            "CUDA cache",
            lambda rows: rows[0].__setitem__(
                "cuda_cache_emptied",
                False,
            ),
            "CUDA cache",
        ),
    )
    for label, mutate, fragment in mutations:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir, source_root = _build_run(Path(temporary_directory))
            evidence_path = run_dir / "rank_evidence.json"
            rows = json.loads(evidence_path.read_text())
            mutate(rows)
            evidence_path.write_bytes(_canonical(rows) + b"\n")
            _resign_artifact(run_dir, "rank_evidence.json")
            _expect_error(
                lambda: verifier.verify_run(
                    run_dir,
                    source_root=source_root,
                ),
                fragment,
            )


def test_verifier_rejects_tensor_metric_source_and_inventory_tamper():
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir, source_root = _build_run(Path(temporary_directory))
        native_path = run_dir / "native_rank0_logits.pt"
        native = torch.load(
            native_path,
            map_location="cpu",
            weights_only=True,
        )
        native["p17"][-1] = -50.0
        torch.save(native, native_path)
        _resign_artifact(run_dir, "native_rank0_logits.pt")
        _expect_error(
            lambda: verifier.verify_run(run_dir, source_root=source_root),
            "comparison",
        )

    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir, source_root = _build_run(Path(temporary_directory))
        result_path = run_dir / "tp4_real_root_logit_correctness.json"
        result = json.loads(result_path.read_text())
        result["comparisons"][0]["max_abs_diff"] = 123.0
        result_path.write_bytes(_canonical(result) + b"\n")
        _resign_artifact(
            run_dir,
            "tp4_real_root_logit_correctness.json",
        )
        _expect_error(
            lambda: verifier.verify_run(run_dir, source_root=source_root),
            "comparison",
        )

    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir, source_root = _build_run(Path(temporary_directory))
        result_path = run_dir / "tp4_real_root_logit_correctness.json"
        result = json.loads(result_path.read_text())
        result["comparisons"][0]["cosine_similarity"] = 0.5
        result_path.write_bytes(_canonical(result) + b"\n")
        _resign_artifact(
            run_dir,
            "tp4_real_root_logit_correctness.json",
        )
        _expect_error(
            lambda: verifier.verify_run(run_dir, source_root=source_root),
            "comparison",
        )

    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir, source_root = _build_run(Path(temporary_directory))
        (source_root / "gate/contract.py").write_text("drift\n")
        _expect_error(
            lambda: verifier.verify_run(run_dir, source_root=source_root),
            "source hash",
        )

    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir, source_root = _build_run(Path(temporary_directory))
        (run_dir / "extra.txt").write_text("forbidden")
        _expect_error(
            lambda: verifier.verify_run(run_dir, source_root=source_root),
            "inventory",
        )


def test_verifier_source_does_not_import_producer_or_tinyvllm():
    source = VERIFIER_PATH.read_text(encoding="utf-8")
    assert "qwen35_tp4_real_root_logit_correctness_preflight" not in source
    assert "tinyvllm" not in source


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 independent verifier tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
