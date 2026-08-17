from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
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
    "qwen35_tp4_engine_contract_for_verifier_test",
    "qwen35_tp4_engine_correctness_contract.py",
)
verifier = _load(
    "verify_qwen35_tp4_engine_correctness_gate",
    "verify_qwen35_tp4_engine_correctness_gate.py",
)


MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38f"
    "e69982efedaf4a3e631ad0b14aad7dd0"
)
SOURCE_TREE_SHA256 = "d" * 64


def _write_json(path, value):
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows():
    rows = []
    for scenario, expected in contract.SCENARIOS.items():
        outputs = list(range(expected["generated_tokens"]))
        rows.append({
            "scenario": scenario,
            "engine_class": contract.ENGINE_CLASS,
            "model_runner_class": contract.MODEL_RUNNER_CLASS,
            "rank_inventory": [0, 1, 2, 3],
            "ack_ranks": [1, 2, 3],
            "scheduler_steps": expected["scheduler_steps"],
            "model_runner_calls": expected["model_runner_calls"],
            "output_token_ids": outputs,
            "reference_output_token_ids": list(outputs),
            "publication_commits": expected["publication_commits"],
            "restore_hits": expected["restore_hits"],
            "restore_misses": expected["restore_misses"],
            "release_events": expected["release_events"],
            "cache_entries_after": expected["cache_entries_after"],
            "cache_identity_match": True,
            "process_group_destroyed": True,
            "rank_exit_codes": [0, 0, 0, 0],
            "owned_children_remaining": [],
        })
    return rows


def _fixture(root):
    run_dir = Path(root)
    run_dir.mkdir(parents=True)
    rows = _rows()
    _write_json(
        run_dir / "engine_correctness.json",
        {
            "schema_version": contract.SCHEMA_VERSION,
            "classification": "PASS",
            "model_manifest_sha256": MODEL_MANIFEST_SHA256,
            "rows": rows,
        },
    )
    _write_json(
        run_dir / "scheduler_observations.json",
        [{
            "scenario": row["scenario"],
            "scheduler_steps": row["scheduler_steps"],
            "model_runner_calls": row["model_runner_calls"],
            "output_token_ids": row["output_token_ids"],
        } for row in rows],
    )
    _write_json(
        run_dir / "rank_events.json",
        [{
            "scenario": row["scenario"],
            "rank_inventory": row["rank_inventory"],
            "ack_ranks": row["ack_ranks"],
            "process_group_destroyed": (
                row["process_group_destroyed"]
            ),
            "rank_exit_codes": row["rank_exit_codes"],
            "owned_children_remaining": (
                row["owned_children_remaining"]
            ),
        } for row in rows],
    )
    _write_json(
        run_dir / "source_manifest.json",
        {
            "schema_version": contract.SCHEMA_VERSION,
            "source_tree_sha256": SOURCE_TREE_SHA256,
            "model_manifest_sha256": MODEL_MANIFEST_SHA256,
            "files": {
                name: _sha256(run_dir / name)
                for name in contract.ARTIFACT_NAMES[:-1]
            },
        },
    )
    return run_dir


def _expect_invalid(mutator, fragment):
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _fixture(Path(temporary) / "run")
        mutator(run_dir)
        try:
            verifier.verify_run(run_dir)
        except verifier.VerificationError as error:
            assert fragment in str(error), str(error)
        else:
            raise AssertionError(f"tamper accepted: {fragment}")


def test_complete_exact_four_fixture_passes_and_writes_external_result():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dir = _fixture(root / "run")
        output = root / "verification.json"
        result = verifier.verify_and_write(
            run_dir,
            output_path=output,
        )

        assert result["classification"] == "PASS"
        assert result["checks"]["scenario_count"] == 6
        assert result["checks"]["restore_hits"] == 1
        assert result["checks"]["restore_misses"] == 3
        assert json.loads(output.read_text()) == result
        assert set(path.name for path in run_dir.iterdir()) == set(
            contract.ARTIFACT_NAMES
        )


def test_inventory_hash_and_symlink_tamper_is_rejected():
    _expect_invalid(
        lambda run_dir: (run_dir / "extra.json").write_text("{}\n"),
        "artifact inventory",
    )
    _expect_invalid(
        lambda run_dir: (
            run_dir / "scheduler_observations.json"
        ).write_text("{}\n"),
        "artifact hash",
    )

    def symlink(run_dir):
        original = run_dir / "rank_events.json"
        target = run_dir / "rank_events.real.json"
        original.rename(target)
        original.symlink_to(target.name)

    _expect_invalid(symlink, "regular file")


def test_resigned_scheduler_or_cleanup_tamper_is_rejected():
    def scheduler(run_dir):
        result_path = run_dir / "engine_correctness.json"
        result = json.loads(result_path.read_text())
        result["rows"][2]["scheduler_steps"] += 1
        _write_json(result_path, result)
        observations_path = run_dir / "scheduler_observations.json"
        observations = json.loads(observations_path.read_text())
        observations[2]["scheduler_steps"] += 1
        _write_json(observations_path, observations)
        source_path = run_dir / "source_manifest.json"
        source = json.loads(source_path.read_text())
        source["files"]["engine_correctness.json"] = _sha256(
            result_path
        )
        source["files"]["scheduler_observations.json"] = _sha256(
            observations_path
        )
        _write_json(source_path, source)

    _expect_invalid(scheduler, "classification")

    def cleanup(run_dir):
        result_path = run_dir / "engine_correctness.json"
        result = json.loads(result_path.read_text())
        result["rows"][-1]["rank_exit_codes"][-1] = 1
        _write_json(result_path, result)
        ranks_path = run_dir / "rank_events.json"
        ranks = json.loads(ranks_path.read_text())
        ranks[-1]["rank_exit_codes"][-1] = 1
        _write_json(ranks_path, ranks)
        source_path = run_dir / "source_manifest.json"
        source = json.loads(source_path.read_text())
        source["files"]["engine_correctness.json"] = _sha256(
            result_path
        )
        source["files"]["rank_events.json"] = _sha256(ranks_path)
        _write_json(source_path, source)

    _expect_invalid(cleanup, "classification")


def test_external_verification_refuses_internal_or_existing_path():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dir = _fixture(root / "run")
        existing = root / "existing.json"
        existing.write_text("{}\n")
        for output in (
            run_dir / "verification.json",
            existing,
        ):
            try:
                verifier.verify_and_write(
                    run_dir,
                    output_path=output,
                )
            except ValueError as error:
                assert (
                    "outside" in str(error)
                    or "already exists" in str(error)
                )
            else:
                raise AssertionError("unsafe output path was accepted")


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 Engine correctness verifier tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
