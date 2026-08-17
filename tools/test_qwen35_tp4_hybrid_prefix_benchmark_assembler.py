from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name, filename):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load(
    "qwen35_tp4_hybrid_prefix_contract_for_assembler_test",
    "qwen35_tp4_hybrid_prefix_benchmark_contract.py",
)
verifier_fixture = _load(
    "qwen35_tp4_hybrid_prefix_verifier_fixture_for_assembler_test",
    "test_verify_qwen35_tp4_hybrid_prefix_benchmark.py",
)
verifier = verifier_fixture.verifier
assembler = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_assembler",
    "qwen35_tp4_hybrid_prefix_benchmark_assembler.py",
)


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
            ) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _inputs(root):
    reference = verifier_fixture._complete_run_dir(
        Path(root) / "reference"
    )
    cases_root = Path(root) / "cases"
    logs_root = Path(root) / "worker-logs"
    case_rows = _read_jsonl(reference / "case_rows.jsonl")
    process_rows = _read_jsonl(reference / "process_rows.jsonl")
    logs = {}
    for case in contract.build_case_matrix():
        case_dir = cases_root / case.case_id
        rows = [
            row for row in case_rows
            if row["case_id"] == case.case_id
        ]
        processes = [
            row for row in process_rows
            if row["case_id"] == case.case_id
        ]
        _write_jsonl(case_dir / "case_rows.jsonl", rows)
        _write_jsonl(case_dir / "process_rows.jsonl", processes)
        _write_json(
            case_dir / "summary.json",
            {
                "schema_version": contract.SCHEMA_VERSION,
                "complete": True,
                "case_id": case.case_id,
                "case_rows": len(rows),
                "process_rows": len(processes),
            },
        )
        for row in rows:
            relative = row["final_logits_path"]
            if relative is not None:
                source = reference / relative
                destination = case_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        log = logs_root / f"{case.case_id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            "QWEN35_TP4_BENCHMARK_WORKER_COMPLETE\n",
            encoding="utf-8",
        )
        logs[case.case_id] = log
    return {
        "reference": reference,
        "cases_root": cases_root,
        "logs": logs,
        "prerequisites": reference / "correctness_prerequisites.json",
        "workload_manifest": reference / "workload_manifest.json",
        "source_manifest": json.loads(
            (reference / "source_manifest.json").read_text(
                encoding="utf-8"
            )
        ),
        "environment": json.loads(
            (reference / "environment.json").read_text(
                encoding="utf-8"
            )
        ),
        "gpu_assignments": json.loads(
            (reference / "gpu_assignments.json").read_text(
                encoding="utf-8"
            )
        ),
        "commands": json.loads(
            (reference / "commands.json").read_text(
                encoding="utf-8"
            )
        )["commands"],
    }


def _assemble(root, inputs):
    output = Path(root) / "assembled"
    result = assembler.assemble_benchmark_run(
        output_dir=output,
        cases_root=inputs["cases_root"],
        correctness_prerequisites_path=inputs["prerequisites"],
        workload_manifest_path=inputs["workload_manifest"],
        source_manifest=inputs["source_manifest"],
        environment=inputs["environment"],
        gpu_assignments=inputs["gpu_assignments"],
        commands=inputs["commands"],
        worker_logs=inputs["logs"],
    )
    return output, result


def test_assembler_publishes_complete_independently_verifiable_run():
    with tempfile.TemporaryDirectory() as temporary:
        inputs = _inputs(temporary)
        output, result = _assemble(temporary, inputs)

        assert result["classification"] == "ASSEMBLED"
        assert result["case_rows"] > len(contract.build_case_matrix())
        assert result["process_rows"] == len(
            contract.build_case_matrix()
        )
        assert set(path.name for path in output.iterdir()) == (
            set(contract.TOP_LEVEL_ARTIFACTS)
            - {"independent_verification.json", "report.md"}
            | set(contract.NESTED_ARTIFACT_DIRECTORIES)
        )
        verification = verifier.verify_run(output)
        assert verification["classification"] == "GO"


def test_assembler_preserves_shared_gpu_assignment_policy():
    with tempfile.TemporaryDirectory() as temporary:
        inputs = _inputs(temporary)
        assignments = inputs["gpu_assignments"]
        assignments["resource_policy"] = "shared-low-utilization"
        assignments["maximum_gpu_utilization_percent"] = 10
        for row in assignments["assignments"]:
            row["utilization_percent"] = 0
            row["compute_processes"] = [{"pid": 1000 + row["rank"]}]

        output, result = _assemble(temporary, inputs)

        assert result["classification"] == "ASSEMBLED"
        assert json.loads(
            (output / "gpu_assignments.json").read_text(
                encoding="utf-8",
            )
        ) == assignments


def test_assembler_rejects_missing_case_without_publishing():
    with tempfile.TemporaryDirectory() as temporary:
        inputs = _inputs(temporary)
        missing = contract.build_case_matrix()[-1].case_id
        shutil.rmtree(inputs["cases_root"] / missing)
        output = Path(temporary) / "assembled"
        try:
            _assemble(temporary, inputs)
        except ValueError as error:
            assert "case inventory" in str(error), str(error)
        else:
            raise AssertionError("missing case was accepted")
        assert not output.exists()


def test_assembler_rejects_traceback_log_without_publishing():
    with tempfile.TemporaryDirectory() as temporary:
        inputs = _inputs(temporary)
        case_id = contract.build_case_matrix()[0].case_id
        inputs["logs"][case_id].write_text(
            "Traceback (most recent call last)\n",
            encoding="utf-8",
        )
        output = Path(temporary) / "assembled"
        try:
            _assemble(temporary, inputs)
        except ValueError as error:
            assert "worker log" in str(error), str(error)
        else:
            raise AssertionError("traceback log was accepted")
        assert not output.exists()


def test_assembler_rejects_case_provenance_drift_without_publishing():
    with tempfile.TemporaryDirectory() as temporary:
        inputs = _inputs(temporary)
        case_id = contract.build_case_matrix()[0].case_id
        path = inputs["cases_root"] / case_id / "case_rows.jsonl"
        rows = _read_jsonl(path)
        rows[0]["source_tree_sha256"] = "f" * 64
        _write_jsonl(path, rows)
        output = Path(temporary) / "assembled"
        try:
            _assemble(temporary, inputs)
        except ValueError as error:
            assert "provenance" in str(error), str(error)
        else:
            raise AssertionError("provenance drift was accepted")
        assert not output.exists()


def test_assembler_cli_consumes_explicit_metadata_files():
    with tempfile.TemporaryDirectory() as temporary:
        inputs = _inputs(temporary)
        metadata = Path(temporary) / "metadata"
        source_manifest = metadata / "source_manifest.json"
        environment = metadata / "environment.json"
        gpu_assignments = metadata / "gpu_assignments.json"
        commands = metadata / "commands.json"
        worker_logs = metadata / "worker_logs.json"
        _write_json(source_manifest, inputs["source_manifest"])
        _write_json(environment, inputs["environment"])
        _write_json(gpu_assignments, inputs["gpu_assignments"])
        _write_json(
            commands,
            {
                "schema_version": contract.SCHEMA_VERSION,
                "commands": inputs["commands"],
            },
        )
        _write_json(
            worker_logs,
            {
                "schema_version": contract.SCHEMA_VERSION,
                "worker_logs": {
                    name: str(path)
                    for name, path in inputs["logs"].items()
                },
            },
        )
        output = Path(temporary) / "assembled"
        result = assembler.main([
            "--output-dir", str(output),
            "--cases-root", str(inputs["cases_root"]),
            "--correctness-prerequisites",
            str(inputs["prerequisites"]),
            "--workload-manifest",
            str(inputs["workload_manifest"]),
            "--source-manifest", str(source_manifest),
            "--environment", str(environment),
            "--gpu-assignments", str(gpu_assignments),
            "--commands", str(commands),
            "--worker-logs", str(worker_logs),
        ])

        assert result == 0
        assert verifier.verify_run(output)["classification"] == "GO"


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 hybrid-prefix benchmark assembler tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
