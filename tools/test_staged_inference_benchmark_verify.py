"""Dependency-light tests for the staged benchmark independent verifier.

Run:
    python3 tools/test_staged_inference_benchmark_verify.py
"""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import staged_inference_benchmark_gate as gate
from tools import staged_inference_benchmark_verify as verifier
from tools import test_staged_inference_benchmark_gate as fixtures


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _source_fixture(root: Path) -> tuple[dict, Path, Path]:
    source_root = root / "source-fixture"
    records = []
    source_paths = sorted([
        (
            f"{owned_root}/__init__.py"
            if owned_root == "tinyvllm"
            else owned_root
        )
        for owned_root in gate.OWNED_SOURCE_ROOTS
    ])
    for ordinal, relative in enumerate(source_paths):
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"synthetic source {ordinal}: {relative}\n".encode()
        path.write_bytes(payload)
        records.append({
            "path": relative,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    patch_path = root / "source.patch"
    patch_path.write_bytes(b"")
    evidence = {
        "schema_version": 1,
        "base_commit": "1" * 40,
        "local_head": "1" * 40,
        "tracking_head": "1" * 40,
        "dirty": False,
        "patch_path": "source.patch",
        "patch_sha256": hashlib.sha256(b"").hexdigest(),
        "patch_size_bytes": 0,
        "owned_roots": list(gate.OWNED_SOURCE_ROOTS),
        "files": records,
        "tree_sha256": _canonical_sha256(records),
    }
    return evidence, source_root, patch_path


def _attach_source_snapshot(
    run_dir: Path,
    source_root: Path,
    patch_path: Path,
) -> None:
    with tarfile.open(run_dir / "source_snapshot.tar", "w") as archive:
        for path in sorted(source_root.rglob("*")):
            if path.is_file():
                archive.add(
                    path,
                    arcname=path.relative_to(source_root).as_posix(),
                    recursive=False,
                )
    shutil.copyfile(patch_path, run_dir / "source.patch")


def _stage1_environment(*, gate_name: str) -> dict:
    environment = fixtures._environment_evidence(gate_name=gate_name)
    environment["gpu_inventory"] = environment["gpu_inventory"][:1]
    environment["selected_gpu_indices"] = [0]
    return environment


def _prefix_correctness_rows() -> list[dict]:
    logit_diff = {
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "reference_argmax": 42,
        "candidate_argmax": 42,
        "argmax_match": True,
        "within_tolerance": True,
    }
    rows = [{
        "case": "cpu_collision_and_lifecycle_preflight",
        "state": "preflight",
        "command": ["python3", "tools/test_chunked_prefill.py"],
        "returncode": 0,
        "stdout": "chunked prefill tests passed",
        "stderr": "",
        "correct": True,
    }]
    for prompt_tokens in (255, 256, 257, 512, 513):
        cached_tokens = ((prompt_tokens - 1) // 256) * 256
        rows.append({
            "case": f"repeat_{prompt_tokens}",
            "state": "warm",
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "query_tokens": prompt_tokens - cached_tokens,
            "token_id": 42,
            "decoded": "token",
            "logit_diff": deepcopy(logit_diff),
            "expected_reusable_tokens": cached_tokens,
            "correct": True,
        })
    for case in (
        "same_batch_p_q_p_first",
        "same_batch_p_q_p_middle",
        "same_batch_p_q_p",
    ):
        row = {
            "case": case,
            "state": "same_batch",
            "prompt_tokens": 513,
            "cached_tokens": 0,
            "query_tokens": 513,
            "token_id": 42,
            "decoded": "token",
            "logit_diff": deepcopy(logit_diff),
            "correct": True,
        }
        if case == "same_batch_p_q_p":
            row["batch_q_logit_diff"] = {
                **deepcopy(logit_diff),
                "max_abs": 1.0,
                "mean_abs": 0.1,
                "candidate_argmax": 43,
                "argmax_match": False,
                "within_tolerance": False,
            }
        rows.append(row)
    rows.extend([
        {
            "case": "shared_prefix_different_suffix",
            "state": "warm",
            "prompt_tokens": 320,
            "cached_tokens": 256,
            "query_tokens": 64,
            "token_id": 42,
            "decoded": "token",
            "logit_diff": deepcopy(logit_diff),
            "expected_reusable_tokens": 256,
            "correct": True,
        },
        {
            "case": "cache_cleared",
            "state": "cache_cleared",
            "prompt_tokens": 320,
            "cached_tokens": 0,
            "query_tokens": 320,
            "token_id": 42,
            "decoded": "token",
            "logit_diff": deepcopy(logit_diff),
            "expected_reusable_tokens": 0,
            "correct": True,
        },
    ])
    return rows


def write_complete_prefix_bundle(
    root: Path,
    *,
    incorrect_correctness_case: str | None = None,
) -> Path:
    run_dir = root / "prefix-run"
    source, source_root, patch_path = _source_fixture(
        root / "prefix-source"
    )
    manifest = gate.initialize_run(
        run_dir=run_dir,
        run_tag="synthetic-qwen3-06b-prefix",
        gate_name="prefix",
        model_tier="qwen3-0.6b",
        source_evidence=source,
        environment_evidence=_stage1_environment(gate_name="prefix"),
    )
    _attach_source_snapshot(run_dir, source_root, patch_path)
    fixtures._populate_prefix_output(
        run_dir,
        manifest,
        incorrect_correctness_case=incorrect_correctness_case,
    )
    case_id = manifest["case_order"][0]
    correctness_rows = _prefix_correctness_rows()
    if incorrect_correctness_case is not None:
        matching_rows = [
            row
            for row in correctness_rows
            if row.get("case") == incorrect_correctness_case
        ]
        assert len(matching_rows) == 1
        matching_rows[0]["correct"] = False
        if "logit_diff" in matching_rows[0]:
            matching_rows[0]["logit_diff"].update({
                "max_abs": 0.375,
                "mean_abs": 0.06396400183439255,
                "within_tolerance": False,
            })
    _write_jsonl(
        run_dir
        / "cases"
        / case_id
        / "output"
        / "prefix_correctness_rows.jsonl",
        correctness_rows,
    )
    process_path = (
        run_dir / "cases" / case_id / "process.json"
    )
    process = _read_json(process_path)
    process.update({"master_port": 31001, "distributed_port": 31002})
    _write_json(process_path, process)
    gate.finalize_run(run_dir)
    return run_dir


def write_complete_chunked_bundle(root: Path) -> Path:
    run_dir = root / "chunked-run"
    source, source_root, patch_path = _source_fixture(
        root / "chunked-source"
    )
    manifest = gate.initialize_run(
        run_dir=run_dir,
        run_tag="synthetic-qwen3-06b-chunked",
        gate_name="chunked",
        model_tier="qwen3-0.6b",
        source_evidence=source,
        environment_evidence=_stage1_environment(gate_name="chunked"),
    )
    _attach_source_snapshot(run_dir, source_root, patch_path)
    fixtures._populate_chunked_outputs(run_dir, manifest)
    for ordinal, case_id in enumerate(manifest["case_order"]):
        process_path = run_dir / "cases" / case_id / "process.json"
        process = _read_json(process_path)
        process.update({
            "master_port": 32000 + ordinal * 2,
            "distributed_port": 32001 + ordinal * 2,
        })
        _write_json(process_path, process)
        result_path = (
            run_dir / "cases" / case_id / "output" / "case_result.json"
        )
        result = _read_json(result_path)
        result.update({
            "request_count": 104,
            "step_count": 1,
            "error": None,
        })
        _write_json(result_path, result)
    gate.finalize_run(run_dir)
    return run_dir


def refresh_artifact_hashes(run_dir: Path) -> None:
    hashes = {}
    for path in sorted(run_dir.rglob("*")):
        if (
            not path.is_file()
            or path.name == "artifact_hashes.json"
            or path.name.endswith(".tmp")
        ):
            continue
        hashes[path.relative_to(run_dir).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    _write_json(run_dir / "artifact_hashes.json", hashes)


def tamper_source_hash(run_dir: Path) -> None:
    with TemporaryDirectory() as temporary:
        source_root = Path(temporary) / "source"
        source_root.mkdir()
        with tarfile.open(
            run_dir / "source_snapshot.tar",
            "r:",
        ) as archive:
            archive.extractall(source_root)
        source_file = next(
            path for path in sorted(source_root.rglob("*"))
            if path.is_file()
        )
        source_file.write_bytes(source_file.read_bytes() + b"tamper\n")
        with tarfile.open(
            run_dir / "source_snapshot.tar",
            "w",
        ) as archive:
            for path in sorted(source_root.rglob("*")):
                if path.is_file():
                    archive.add(
                        path,
                        arcname=path.relative_to(source_root).as_posix(),
                        recursive=False,
                    )


def tamper_nonempty_source_patch(run_dir: Path) -> None:
    payload = b"diff --git a/file b/file\n"
    (run_dir / "source.patch").write_bytes(payload)
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["source_evidence"]["patch_size_bytes"] = len(payload)
    manifest["source_evidence"]["patch_sha256"] = hashlib.sha256(
        payload
    ).hexdigest()
    _write_json(manifest_path, manifest)
    (run_dir / "manifest.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )


def tamper_workload(run_dir: Path) -> None:
    path = run_dir / "workload_manifest.jsonl"
    rows = _read_jsonl(path)
    rows[0]["tampered"] = True
    _write_jsonl(path, rows)


def tamper_stage1_gpu_count(run_dir: Path) -> None:
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    environment = manifest["environment_evidence"]
    environment["gpu_inventory"] = [
        {
            "index": index,
            "uuid": f"GPU-{index}",
            "name": "NVIDIA H100 80GB HBM3",
        }
        for index in range(4)
    ]
    environment["selected_gpu_indices"] = [0, 1, 2, 3]
    manifest["environment_sha256"] = _canonical_sha256(environment)
    _write_json(manifest_path, manifest)

    resolved_path = run_dir / "resolved_config.json"
    resolved = _read_json(resolved_path)
    resolved["environment"] = deepcopy(environment)
    _write_json(resolved_path, resolved)
    receipt_path = run_dir / "primary_verification_receipt.json"
    receipt = _read_json(receipt_path)
    receipt["environment_sha256"] = manifest["environment_sha256"]
    _write_json(receipt_path, receipt)
    (run_dir / "manifest.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )


def tamper_output_token(run_dir: Path) -> None:
    path = run_dir / "request_timeline.jsonl"
    rows = _read_jsonl(path)
    measured = next(row for row in rows if row.get("warmup") is False)
    measured["output_token_ids"][0] += 1
    _write_jsonl(path, rows)


def tamper_cached_tokens(run_dir: Path) -> None:
    path = run_dir / "cache_trace.jsonl"
    rows = _read_jsonl(path)
    rows[0]["cached_prompt_tokens"] += 1
    _write_jsonl(path, rows)


def tamper_unreconciled_prefix_cache_cost(run_dir: Path) -> None:
    manifest = _read_json(run_dir / "run_manifest.json")
    case_id = manifest["case_order"][0]
    paths = (
        run_dir
        / "cases"
        / case_id
        / "output"
        / "prefix_performance_rows.jsonl",
        run_dir / "scheduler_trace.jsonl",
        run_dir / "case_rows.jsonl",
    )
    for path in paths:
        rows = _read_jsonl(path)
        rows[0]["cache_clear_host_ns"] += 1
        _write_jsonl(path, rows)


def tamper_strip_prefix_correctness_evidence(run_dir: Path) -> None:
    manifest = _read_json(run_dir / "run_manifest.json")
    case_id = manifest["case_order"][0]
    paths = (
        run_dir
        / "cases"
        / case_id
        / "output"
        / "prefix_correctness_rows.jsonl",
        run_dir / "request_timeline.jsonl",
    )
    for path in paths:
        rows = _read_jsonl(path)
        _write_jsonl(
            path,
            [
                {
                    "case": row["case"],
                    "state": row["state"],
                    "correct": True,
                }
                for row in rows
            ],
        )


def tamper_ttft_summary(run_dir: Path) -> None:
    path = run_dir / "summary.json"
    summary = _read_json(path)
    benefit = summary["benefit"]
    key = next(iter(benefit))
    benefit[key] = float(benefit[key]) + 0.01
    _write_json(path, summary)


def tamper_cuda_reserved(run_dir: Path) -> None:
    path = run_dir / "memory_trace.jsonl"
    rows = _read_jsonl(path)
    rows[0]["cuda_peak_reserved_bytes"] += 100
    _write_jsonl(path, rows)


def tamper_scheduler_step_index(run_dir: Path) -> None:
    manifest = _read_json(run_dir / "run_manifest.json")
    case_id = manifest["case_order"][0]
    case_path = (
        run_dir / "cases" / case_id / "output" / "scheduler_trace.jsonl"
    )
    case_rows = _read_jsonl(case_path)
    case_rows[0]["step_index"] = -1
    _write_jsonl(case_path, case_rows)

    merged_path = run_dir / "scheduler_trace.jsonl"
    merged_rows = _read_jsonl(merged_path)
    merged_rows[0]["step_index"] = -1
    _write_jsonl(merged_path, merged_rows)


def tamper_lifecycle_order(run_dir: Path) -> None:
    manifest = _read_json(run_dir / "run_manifest.json")
    case_id = manifest["case_order"][0]
    case_path = (
        run_dir / "cases" / case_id / "output" / "request_timeline.jsonl"
    )
    case_rows = _read_jsonl(case_path)
    case_rows[0]["actual_arrival_ns"] = (
        case_rows[0]["scheduled_arrival_ns"] - 1
    )
    _write_jsonl(case_path, case_rows)

    merged_path = run_dir / "request_timeline.jsonl"
    merged_rows = _read_jsonl(merged_path)
    merged_rows[0]["actual_arrival_ns"] = (
        merged_rows[0]["scheduled_arrival_ns"] - 1
    )
    _write_jsonl(merged_path, merged_rows)


def tamper_arrival_offset_without_changing_metrics(run_dir: Path) -> None:
    manifest = _read_json(run_dir / "run_manifest.json")
    case_id = manifest["case_order"][0]
    case_path = (
        run_dir / "cases" / case_id / "output" / "request_timeline.jsonl"
    )
    case_rows = _read_jsonl(case_path)
    for field in (
        "scheduled_arrival_ns",
        "actual_arrival_ns",
        "first_scheduled_ns",
        "first_token_ns",
        "completion_ns",
    ):
        case_rows[0][field] += 1
    case_rows[0]["token_timestamps_ns"] = [
        value + 1
        for value in case_rows[0]["token_timestamps_ns"]
    ]
    _write_jsonl(case_path, case_rows)

    merged_path = run_dir / "request_timeline.jsonl"
    merged_rows = _read_jsonl(merged_path)
    for field in (
        "scheduled_arrival_ns",
        "actual_arrival_ns",
        "first_scheduled_ns",
        "first_token_ns",
        "completion_ns",
    ):
        merged_rows[0][field] += 1
    merged_rows[0]["token_timestamps_ns"] = [
        value + 1
        for value in merged_rows[0]["token_timestamps_ns"]
    ]
    _write_jsonl(merged_path, merged_rows)


def tamper_case_spec(run_dir: Path) -> None:
    manifest = _read_json(run_dir / "run_manifest.json")
    case_id = manifest["case_order"][0]
    path = run_dir / manifest["case_specs"][case_id]
    case = _read_json(path)
    case["model"] = "/models/not-the-bound-checkpoint"
    _write_json(path, case)


def tamper_duplicate_port(run_dir: Path) -> None:
    manifest = _read_json(run_dir / "run_manifest.json")
    first, second = manifest["case_order"][:2]
    first_process = _read_json(run_dir / "cases" / first / "process.json")
    second_path = run_dir / "cases" / second / "process.json"
    second_process = _read_json(second_path)
    second_process["master_port"] = first_process["master_port"]
    _write_json(second_path, second_process)


def tamper_out_of_range_port(run_dir: Path) -> None:
    manifest = _read_json(run_dir / "run_manifest.json")
    case_id = manifest["case_order"][0]
    path = run_dir / "cases" / case_id / "process.json"
    process = _read_json(path)
    process["master_port"] = 70_000
    _write_json(path, process)


def tamper_reordered_timeline(run_dir: Path) -> None:
    manifest = _read_json(run_dir / "run_manifest.json")
    case_id = manifest["case_order"][0]
    case_path = (
        run_dir / "cases" / case_id / "output" / "request_timeline.jsonl"
    )
    case_rows = _read_jsonl(case_path)
    case_rows[0], case_rows[1] = case_rows[1], case_rows[0]
    _write_jsonl(case_path, case_rows)

    merged_path = run_dir / "request_timeline.jsonl"
    merged_rows = _read_jsonl(merged_path)
    merged_rows[0], merged_rows[1] = merged_rows[1], merged_rows[0]
    _write_jsonl(merged_path, merged_rows)


def tamper_incomplete_case_result(run_dir: Path) -> None:
    manifest = _read_json(run_dir / "run_manifest.json")
    case_id = manifest["case_order"][0]
    path = run_dir / "cases" / case_id / "output" / "case_result.json"
    result = _read_json(path)
    for field in ("request_count", "step_count", "error"):
        result.pop(field)
    _write_json(path, result)


def tamper_unsafe_source_member(run_dir: Path) -> None:
    payload = b"escape\n"
    member = tarfile.TarInfo("../escape")
    member.size = len(payload)
    with tarfile.open(run_dir / "source_snapshot.tar", "a") as archive:
        archive.addfile(member, io.BytesIO(payload))


def tamper_report(run_dir: Path) -> None:
    path = run_dir / "report.md"
    path.write_text(path.read_text(encoding="utf-8") + "tamper\n")


def tamper_unhashed_temporary_file(run_dir: Path) -> None:
    (run_dir / "unverified.tmp").write_text(
        "not covered by artifact_hashes.json\n",
        encoding="utf-8",
    )


def tamper_symlink_artifact(run_dir: Path) -> None:
    (run_dir / "linked-summary.json").symlink_to("summary.json")


def truncate_jsonl(run_dir: Path) -> None:
    path = run_dir / "case_rows.jsonl"
    path.write_bytes(path.read_bytes().rstrip(b"\n"))


def duplicate_case(run_dir: Path) -> None:
    path = run_dir / "case_rows.jsonl"
    rows = _read_jsonl(path)
    rows[-1] = json.loads(json.dumps(rows[0]))
    _write_jsonl(path, rows)


def test_verifier_uses_only_standard_library_and_artifacts():
    tree = ast.parse(
        (REPO_ROOT / "tools" / "staged_inference_benchmark_verify.py")
        .read_text(encoding="utf-8")
    )
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    for forbidden in (
        "tools.staged_inference_benchmark_gate",
        "tools.profile_prefix_cache",
        "tools.arrival_load_gate",
    ):
        assert forbidden not in imported


def test_source_snapshot_size_mismatch_is_rejected_before_extraction():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        archive_path = root / "source.tar"
        payload = b"oversized"
        member = tarfile.TarInfo("tools/arrival_load_driver.py")
        member.size = len(payload)
        with tarfile.open(archive_path, "w") as archive:
            archive.addfile(member, io.BytesIO(payload))
        destination = root / "extracted"
        destination.mkdir()

        try:
            verifier._safe_extract_source_snapshot(
                archive_path,
                destination,
                expected_files={"tools/arrival_load_driver.py": 1},
            )
        except ValueError:
            pass
        else:
            raise AssertionError("source size mismatch must fail")
        assert not any(destination.rglob("*"))


def test_verifier_rebuilds_complete_prefix_bundle():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dir = write_complete_prefix_bundle(root)
        controller = root / "prefix-controller"

        result = verifier.verify_run(run_dir, controller)

        assert result["classification"] == "PREFIX_CACHE_GO"
        receipt = _read_json(controller / "verification_receipt.json")
        assert receipt["status"] == "PASS"
        assert receipt["classification"] == "PREFIX_CACHE_GO"
        assert (controller / "summary.json").read_bytes() == (
            run_dir / "summary.json"
        ).read_bytes()
        assert (controller / "report.md").read_bytes() == (
            run_dir / "report.md"
        ).read_bytes()
        assert (controller / "verify.exitcode").read_text() == "0\n"


def test_verifier_rebuilds_complete_chunked_bundle():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dir = write_complete_chunked_bundle(root)
        result = verifier.verify_run(
            run_dir,
            root / "chunked-controller",
        )
        assert result["classification"] == "FAIR_CHUNKED_GO"


def test_verifier_rebuilds_complete_starvation_evidence():
    workload = gate.contract.build_chunked_workload()
    case = gate.contract.build_chunked_case_matrix(
        model_tier="qwen3-0.6b"
    )[1]
    timeline = fixtures._chunked_timeline(
        workload,
        policy=case["policy"],
    )
    for row in timeline[8:10]:
        row["first_scheduled_ns"] = (
            row["scheduled_arrival_ns"]
            + row["starvation_deadline_ns"]
            + 1
        )
        row["first_token_ns"] = row["first_scheduled_ns"] + 10
        row["token_timestamps_ns"] = [
            row["first_token_ns"] + index * 100
            for index in range(len(row["output_token_ids"]))
        ]
        row["completion_ns"] = row["token_timestamps_ns"][-1]
    metrics, _ = verifier._chunked_case_metrics(
        case,
        timeline,
        workload,
        [{"cuda_peak_reserved_bytes": 1_000}],
        {
            "case_id": case["case_id"],
            "status": "INCOMPLETE",
            "error_type": "starved_request",
            "error": "request exceeded starvation deadline",
            "request_count": 104,
            "completed_request_count": 104,
            "step_count": 1,
        },
    )

    assert metrics["complete_lifecycle"] is True
    assert metrics["starved_requests"] == 2


def test_verifier_classifies_complete_incorrect_prefix_evidence():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dir = write_complete_prefix_bundle(
            root,
            incorrect_correctness_case="repeat_513",
        )

        result = verifier.verify_run(
            run_dir,
            root / "prefix-controller",
        )

        assert result["classification"] == (
            "PREFIX_CACHE_INCOMPLETE_OR_INCORRECT"
        )
        assert result["structural_failures"] == []
        assert result["correctness_failures"] == [
            "repeat_513: targeted correctness check failed"
        ]


def test_verifier_thresholds_match_the_frozen_contract():
    prefix = fixtures._complete_prefix_bundle()
    for prefix_tokens in ("1024", "2048"):
        prefix["single"][prefix_tokens]["warm"][
            "median_elapsed_ms"
        ] = 80.0
    for prefix_tokens in ("1024", "2048"):
        prefix["batch"][prefix_tokens]["warm"][
            "median_elapsed_ms"
        ] = 85.0
    assert verifier._classify_prefix(deepcopy(prefix)) == (
        gate.contract.classify_prefix_bundle(deepcopy(prefix))
    )
    prefix["single"]["1024"]["warm"]["median_elapsed_ms"] = 80.1
    assert verifier._classify_prefix(deepcopy(prefix)) == (
        gate.contract.classify_prefix_bundle(deepcopy(prefix))
    )

    baseline = {
        "short_p99_ttft_ns": 100.0,
        "short_p99_itl_ns": 100.0,
        "maximum_decode_gap_ns": 100.0,
        "service_class_p95_completion_ns": {
            bucket: 100.0
            for bucket in gate.SERVICE_CLASS_BUCKETS
        },
        "long_p95_completion_ns": 100.0,
        "request_throughput_rps": 100.0,
        "output_token_throughput_tps": 100.0,
        "peak_cuda_reserved_bytes": 100.0,
        "exact_outputs": True,
        "complete_lifecycle": True,
        "dropped_requests": 0,
        "rejected_requests": 0,
        "truncated_requests": 0,
        "unfinished_requests": 0,
        "starved_requests": 0,
    }
    candidate = deepcopy(baseline)
    candidate.update({
        "short_p99_ttft_ns": 90.0,
        "short_p99_itl_ns": 105.0,
        "maximum_decode_gap_ns": 110.0,
        "long_p95_completion_ns": 110.0,
        "request_throughput_rps": 97.0,
        "output_token_throughput_tps": 97.0,
        "peak_cuda_reserved_bytes": 105.0,
    })
    candidate["service_class_p95_completion_ns"] = {
        bucket: 110.0
        for bucket in gate.SERVICE_CLASS_BUCKETS
    }
    chunked = {
        "artifact_complete": True,
        "repetitions": [
            {
                "repetition": repetition,
                "OFF": deepcopy(baseline),
                "FAIR_CHUNKED": deepcopy(candidate),
            }
            for repetition in range(5)
        ],
    }
    assert verifier._classify_chunked(deepcopy(chunked)) == (
        gate.contract.classify_chunked_bundle(deepcopy(chunked))
    )
    chunked["repetitions"][0]["FAIR_CHUNKED"][
        "short_p99_ttft_ns"
    ] = 91.0
    chunked["repetitions"][1]["FAIR_CHUNKED"][
        "short_p99_ttft_ns"
    ] = 91.0
    assert verifier._classify_chunked(deepcopy(chunked)) == (
        gate.contract.classify_chunked_bundle(deepcopy(chunked))
    )


def test_verifier_rejects_unrehashed_artifact_tamper():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dir = write_complete_prefix_bundle(root)
        tamper_report(run_dir)
        try:
            verifier.verify_run(run_dir, root / "controller")
        except ValueError as error:
            assert "artifact hash mismatch" in str(error)
        else:
            raise AssertionError("unrehased artifact tamper must fail")


def test_verifier_fails_closed_on_rehashed_tamper():
    mutations = (
        (write_complete_prefix_bundle, tamper_source_hash),
        (write_complete_prefix_bundle, tamper_nonempty_source_patch),
        (write_complete_prefix_bundle, tamper_workload),
        (write_complete_prefix_bundle, tamper_stage1_gpu_count),
        (write_complete_chunked_bundle, tamper_output_token),
        (write_complete_prefix_bundle, tamper_cached_tokens),
        (
            write_complete_prefix_bundle,
            tamper_unreconciled_prefix_cache_cost,
        ),
        (
            write_complete_prefix_bundle,
            tamper_strip_prefix_correctness_evidence,
        ),
        (write_complete_prefix_bundle, tamper_ttft_summary),
        (write_complete_prefix_bundle, tamper_cuda_reserved),
        (write_complete_chunked_bundle, tamper_scheduler_step_index),
        (write_complete_chunked_bundle, tamper_lifecycle_order),
        (
            write_complete_chunked_bundle,
            tamper_arrival_offset_without_changing_metrics,
        ),
        (write_complete_chunked_bundle, tamper_case_spec),
        (write_complete_chunked_bundle, tamper_duplicate_port),
        (write_complete_chunked_bundle, tamper_out_of_range_port),
        (write_complete_chunked_bundle, tamper_reordered_timeline),
        (write_complete_chunked_bundle, tamper_incomplete_case_result),
        (write_complete_prefix_bundle, tamper_unsafe_source_member),
        (write_complete_prefix_bundle, tamper_report),
        (write_complete_prefix_bundle, tamper_unhashed_temporary_file),
        (write_complete_prefix_bundle, tamper_symlink_artifact),
        (write_complete_chunked_bundle, truncate_jsonl),
        (write_complete_chunked_bundle, duplicate_case),
    )
    for builder, mutation in mutations:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = builder(root)
            mutation(run_dir)
            refresh_artifact_hashes(run_dir)
            controller = root / "controller"
            try:
                verifier.verify_run(run_dir, controller)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"verifier accepted tamper: {mutation.__name__}"
                )
            assert not controller.exists()


def main():
    test_verifier_uses_only_standard_library_and_artifacts()
    test_source_snapshot_size_mismatch_is_rejected_before_extraction()
    test_verifier_rebuilds_complete_prefix_bundle()
    test_verifier_rebuilds_complete_chunked_bundle()
    test_verifier_rebuilds_complete_starvation_evidence()
    test_verifier_classifies_complete_incorrect_prefix_evidence()
    test_verifier_thresholds_match_the_frozen_contract()
    test_verifier_rejects_unrehashed_artifact_tamper()
    test_verifier_fails_closed_on_rehashed_tamper()
    print("staged inference benchmark verifier tests passed")


if __name__ == "__main__":
    main()
