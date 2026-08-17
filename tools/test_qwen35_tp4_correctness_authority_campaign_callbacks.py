from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace
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


callbacks_module = _load(
    "qwen35_tp4_correctness_authority_campaign_callbacks",
    "qwen35_tp4_correctness_authority_campaign_callbacks.py",
)


@dataclass(frozen=True)
class FakeRun:
    name: str
    run_tag: str
    authority_dir: Path
    plan_path: Path
    consumed_authorization_path: Path
    receipt_path: Path


@dataclass(frozen=True)
class FakeInput:
    name: str
    run_tag: str
    source_tree_sha256: str
    artifact_path: Path
    artifact_sha256: str
    independent_verification_path: Path
    independent_verification_sha256: str
    provenance_path: Path
    provenance_sha256: str


def _dependencies(events, root):
    def child_executor(kind):
        return SimpleNamespace(
            execute_verified_plan_file=lambda **kwargs: (
                events.append((kind, kwargs)),
                {"classification": "PASS"},
            )[1]
        )

    adapter = SimpleNamespace(
        RealAuthorityRun=FakeRun,
        adapt_real_authorities=lambda **kwargs: [
            FakeInput(
                name=run.name,
                run_tag=run.run_tag,
                source_tree_sha256="a" * 64,
                artifact_path=root / f"{run.name}.artifact.json",
                artifact_sha256="b" * 64,
                independent_verification_path=(
                    root / f"{run.name}.verification.json"
                ),
                independent_verification_sha256="c" * 64,
                provenance_path=root / f"{run.name}.provenance.json",
                provenance_sha256="d" * 64,
            )
            for run in kwargs["runs"]
        ],
    )

    def build_bundle(*, output_dir, authorities):
        output = Path(output_dir)
        output.mkdir()
        prerequisite = output / "correctness_prerequisites.json"
        prerequisite.write_text("{}\n")
        for row in authorities:
            assert isinstance(row, FakeInput)
        return {
            "classification": "PASS",
            "correctness_prerequisites_sha256": "e" * 64,
        }

    return {
        "root_plan": SimpleNamespace(
            verify_remote_execution_plan=lambda path: {"run_tag": "root"}
        ),
        "root_executor": child_executor("root"),
        "root_receipt": SimpleNamespace(
            verify_receipt_files=lambda **kwargs: {
                "classification": "PASS"
            }
        ),
        "cached_plan": SimpleNamespace(
            verify_remote_execution_plan=lambda path: {"run_tag": "cached"}
        ),
        "cached_executor": child_executor("cached"),
        "cached_receipt": SimpleNamespace(
            verify_receipt_files=lambda **kwargs: {
                "classification": "PASS"
            }
        ),
        "engine_plan": SimpleNamespace(
            verify_remote_execution_plan=lambda path: {"run_tag": "engine"}
        ),
        "engine_executor": child_executor("engine"),
        "engine_receipt": SimpleNamespace(
            verify_receipt_files=lambda **kwargs: {
                "classification": "PASS"
            }
        ),
        "adapter": adapter,
        "builder": SimpleNamespace(
            AuthorityInput=FakeInput,
            build_prerequisite_bundle=build_bundle,
        ),
        "contract": SimpleNamespace(
            validate_prerequisites=lambda path: SimpleNamespace(
                classification="PASS",
                authorized=True,
                reasons=(),
            )
        ),
    }


def _child(root, name):
    return {
        "name": name,
        "run_tag": f"{name}-run",
        "authority_dir": str(root / name / "authority"),
        "plan_path": str(root / name / "plan.json"),
        "authorization_path": str(root / name / "authorization.json"),
        "consumed_authorization_path": str(root / name / "consumed.json"),
        "receipt_path": str(root / name / "receipt.json"),
        "failure_path": str(root / name / "failure.json"),
    }


def test_callbacks_require_explicit_runners_and_map_child_execution():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        events = []
        dependencies = _dependencies(events, root)
        try:
            callbacks_module.build_campaign_callbacks(
                command_runner=None,
                root_stage_runner=lambda **kwargs: None,
                dependencies=dependencies,
            )
        except ValueError as error:
            assert "command runner" in str(error), str(error)
        else:
            raise AssertionError("missing command runner was accepted")

        callbacks = callbacks_module.build_campaign_callbacks(
            command_runner=lambda **kwargs: None,
            root_stage_runner=lambda **kwargs: None,
            dependencies=dependencies,
        )
        for name in (
            "tp4_root_logit",
            "cached_continuation",
            "engine_correctness",
        ):
            result = callbacks.child_executors[name](
                child=_child(root, name),
                execution_env={"KRB5CCNAME": "FILE:test"},
            )
            assert result["classification"] == "PASS"
        assert [row[0] for row in events] == [
            "root",
            "cached",
            "engine",
        ]


def test_callbacks_convert_adapter_rows_and_build_inventory():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        callbacks = callbacks_module.build_campaign_callbacks(
            command_runner=lambda **kwargs: None,
            root_stage_runner=lambda **kwargs: None,
            dependencies=_dependencies([], root),
        )
        rows = callbacks.adapt_callback(
            runs=[
                _child(root, name)
                for name in (
                    "tp4_root_logit",
                    "cached_continuation",
                    "engine_correctness",
                )
            ],
            verification_output_dir=root / "verification",
        )
        assert [row["name"] for row in rows] == [
            "tp4_root_logit",
            "cached_continuation",
            "engine_correctness",
        ]
        assert all(isinstance(row["artifact_path"], str) for row in rows)
        result = callbacks.build_callback(
            authorities=rows,
            output_dir=root / "bundle",
        )
        assert result == {
            "classification": "PASS",
            "prerequisite_path": str(
                root / "bundle/correctness_prerequisites.json"
            ),
            "prerequisite_sha256": "e" * 64,
            "owned_files": ["correctness_prerequisites.json"],
        }
        assert callbacks.prerequisite_validator(
            result["prerequisite_path"]
        ) == {
            "classification": "PASS",
            "authorized": True,
            "reasons": [],
        }


def test_default_dependencies_load_in_a_clean_module_namespace():
    dependencies = callbacks_module._default_dependencies()
    assert set(dependencies) == {
        "root_plan",
        "root_executor",
        "root_receipt",
        "cached_plan",
        "cached_executor",
        "cached_receipt",
        "engine_plan",
        "engine_executor",
        "engine_receipt",
        "adapter",
        "builder",
        "contract",
    }


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 correctness authority campaign callback tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
