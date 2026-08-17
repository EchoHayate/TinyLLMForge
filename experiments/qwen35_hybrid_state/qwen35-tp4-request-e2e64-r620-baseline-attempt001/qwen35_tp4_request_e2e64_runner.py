from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "experiments/qwen35_hybrid_state"
VERIFIED_LAUNCHER = (
    EXPERIMENT_ROOT
    / "qwen35-tp4-decode-phase-split-20260811-r631-attempt004"
    / "launch_w2.py"
)
COMMAND_TEMPLATE = (
    EXPERIMENT_ROOT
    / "qwen35-tp4-decode-row-parallel-fp32-20260811-r630-attempt001"
    / "commands.json"
)
TEMPLATE_TAG = (
    "qwen35-tp4-decode-row-parallel-fp32-20260811-r630-attempt001"
)
TEMPLATE_SOURCE_SHA256 = (
    "7a80d62c2c9e71f7899dc397f810427286b330d95da2b63f67839aa98d47b3b3"
)
PREREQUISITES_SHA256 = (
    "35b4bf092d5c4c84746b88ecd88b32bf14357a21d2923336d62653186cf352f8"
)
WORKLOAD_MANIFEST_SHA256 = (
    "71909b825d1a8d162604f6cc3d34ad413b2af6c191425ec007859715a4d084e3"
)
MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0"
)
SELECTED_GPUS = (2, 4, 5, 6)
MINIMUM_FREE_BYTES = 25 * 1024**3
MAXIMUM_UTILIZATION_PERCENT = 10
GENERATED_TOKENS_PER_REQUEST = 64
REQUESTS_PER_CASE = 4
POLICIES = ("recompute", "exact_restore")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    tag: str
    source_tree_sha256: str
    source_tar: Path
    source_tar_sha256: str


SOURCES = {
    "r620": SourceSpec(
        name="r620",
        tag="qwen35-tp4-request-e2e64-r620-baseline-attempt001",
        source_tree_sha256=(
            "a26c543e79a9d4927fd0451d4a287363a677568a1daefe65a2a234a22f5997aa"
        ),
        source_tar=(
            EXPERIMENT_ROOT
            / "qwen35-tp4-decode-internal-profile-20260811-r620-attempt001"
            / "inputs/benchmark_source.tar"
        ),
        source_tar_sha256=(
            "5c39d91203d6c75a487936161bb1bb62e5487b67d4648ea2d464f84be85cd50e"
        ),
    ),
    "r631": SourceSpec(
        name="r631",
        tag="qwen35-tp4-request-e2e64-r631-candidate-attempt001",
        source_tree_sha256=(
            "6f881fae7010cc5f048100b147a72fbf27ffba0f77bc34e2e2e68388a98a2837"
        ),
        source_tar=(
            EXPERIMENT_ROOT
            / "qwen35-tp4-decode-phase-split-20260811-r631-attempt006"
            / "inputs/benchmark_source.tar"
        ),
        source_tar_sha256=(
            "f791f27e807e602f889345d301b72035dcd4a93d55a32adf51fd5eb3eaefb79c"
        ),
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_case_ids() -> set[str]:
    return {
        "w2_long_reuse__warmup__r0__recompute",
        "w2_long_reuse__warmup__r0__exact_restore",
        *{
            f"w2_long_reuse__measured__r{repetition}__{policy}"
            for repetition in range(5)
            for policy in POLICIES
        },
    }


def _replace(value, *, source: SourceSpec):
    if isinstance(value, str):
        return value.replace(
            TEMPLATE_TAG,
            source.tag,
        ).replace(
            TEMPLATE_SOURCE_SHA256,
            source.source_tree_sha256,
        )
    if isinstance(value, list):
        return [
            _replace(item, source=source)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _replace(item, source=source)
            for key, item in value.items()
        }
    return value


def _remove_short_output_arguments(argv: list[str]) -> list[str]:
    result = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--generated-tokens-override":
            if index + 1 >= len(argv):
                raise ValueError(
                    "generated-tokens-override has no value"
                )
            index += 2
            continue
        if argument == "--decode-internal-profile":
            index += 1
            continue
        result.append(argument)
        index += 1
    return result


def _argument_value(argv: list[str], name: str) -> str:
    try:
        index = argv.index(name)
    except ValueError as error:
        raise ValueError(f"missing command argument: {name}") from error
    if index + 1 >= len(argv):
        raise ValueError(f"missing command value: {name}")
    return argv[index + 1]


def validate_source(source: SourceSpec) -> None:
    expected = SOURCES.get(source.name)
    if expected is None or source != expected:
        raise ValueError(
            f"source name {source.name!r} does not match frozen spec"
        )


def build_commands(source: SourceSpec) -> list[dict]:
    validate_source(source)
    commands = json.loads(
        COMMAND_TEMPLATE.read_text(encoding="utf-8")
    )
    commands = _replace(commands, source=source)
    for row in commands:
        row["argv"] = _remove_short_output_arguments(
            list(row["argv"])
        )
    validate_commands(commands, source)
    return commands


def validate_commands(
    commands: list[dict],
    source: SourceSpec,
) -> None:
    validate_source(source)
    if len(commands) != 12:
        raise ValueError("expected exactly 12 commands")
    case_ids = {row.get("case_id") for row in commands}
    if case_ids != _expected_case_ids():
        raise ValueError("command case matrix is not canonical")
    for row in commands:
        case_id = row["case_id"]
        argv = row.get("argv")
        if not isinstance(argv, list):
            raise ValueError(f"invalid argv: {case_id}")
        if "--generated-tokens-override" in argv:
            raise ValueError(
                f"generated-tokens-override remains: {case_id}"
            )
        if "--decode-internal-profile" in argv:
            raise ValueError(
                f"decode-internal-profile remains: {case_id}"
            )
        if "--profile" not in argv:
            raise ValueError(f"profile flag missing: {case_id}")
        if _argument_value(argv, "--workload") != "w2_long_reuse":
            raise ValueError(f"unexpected workload: {case_id}")
        if (
            _argument_value(argv, "--source-tree-sha256")
            != source.source_tree_sha256
        ):
            raise ValueError(f"source mismatch: {case_id}")
        if (
            _argument_value(argv, "--prerequisites-sha256")
            != PREREQUISITES_SHA256
        ):
            raise ValueError(f"prerequisite mismatch: {case_id}")
        if (
            _argument_value(argv, "--workload-manifest-sha256")
            != WORKLOAD_MANIFEST_SHA256
        ):
            raise ValueError(f"workload manifest mismatch: {case_id}")
        if (
            _argument_value(argv, "--model-manifest-sha256")
            != MODEL_MANIFEST_SHA256
        ):
            raise ValueError(f"model manifest mismatch: {case_id}")
        joined = " ".join(argv)
        if source.tag not in joined:
            raise ValueError(f"attempt tag missing: {case_id}")
        if TEMPLATE_TAG in joined:
            raise ValueError(f"template tag remains: {case_id}")
        phase = _argument_value(argv, "--phase")
        policy = _argument_value(argv, "--policy")
        repetition = int(_argument_value(argv, "--repetition"))
        expected_case_id = (
            f"w2_long_reuse__{phase}__r{repetition}__{policy}"
        )
        if expected_case_id != case_id:
            raise ValueError(f"case arguments mismatch: {case_id}")


def build_dry_run_payload(source: SourceSpec) -> dict:
    commands = build_commands(source)
    return {
        "schema_version": "qwen35.tp4-request-e2e64-dry-run.v1",
        "source_name": source.name,
        "run_tag": source.tag,
        "source_tree_sha256": source.source_tree_sha256,
        "source_tar": str(source.source_tar),
        "source_tar_sha256": source.source_tar_sha256,
        "generated_tokens_per_request": (
            GENERATED_TOKENS_PER_REQUEST
        ),
        "requests_per_case": REQUESTS_PER_CASE,
        "selected_gpus": list(SELECTED_GPUS),
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "maximum_utilization_percent": (
            MAXIMUM_UTILIZATION_PERCENT
        ),
        "shared_non_exclusive": True,
        "commands": commands,
    }


def _load_verified_launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "qwen35_request_e2e64_verified_launcher",
        VERIFIED_LAUNCHER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load verified launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_launcher(
    module: ModuleType,
    source: SourceSpec,
    commands: list[dict],
) -> None:
    module.TAG = source.tag
    module.OUTPUT = EXPERIMENT_ROOT / source.tag
    module.REMOTE = f"{module.runner.REMOTE_ROOT}/{source.tag}"
    module.SOURCE = source.source_tree_sha256
    module.SOURCE_TAR = source.source_tar
    module.SOURCE_TAR_SHA = source.source_tar_sha256
    module.build_commands = lambda: commands


def validate_local_inputs(source: SourceSpec) -> None:
    validate_source(source)
    if not VERIFIED_LAUNCHER.is_file():
        raise ValueError("verified launcher is missing")
    if not COMMAND_TEMPLATE.is_file():
        raise ValueError("command template is missing")
    if not source.source_tar.is_file():
        raise ValueError("source tar is missing")
    if _sha256(source.source_tar) != source.source_tar_sha256:
        raise ValueError("source tar SHA mismatch")


def check_freshness(source: SourceSpec) -> dict:
    validate_local_inputs(source)
    local_output = EXPERIMENT_ROOT / source.tag
    module = _load_verified_launcher()
    _configure_launcher(module, source, build_commands(source))
    remote_exists = module.remote_path_exists(module.REMOTE)
    return {
        "source_name": source.name,
        "run_tag": source.tag,
        "local_output": str(local_output),
        "local_exists": local_output.exists(),
        "remote_output": module.REMOTE,
        "remote_exists": remote_exists,
        "fresh": not local_output.exists() and not remote_exists,
    }


def run_source(source: SourceSpec) -> int:
    validate_local_inputs(source)
    commands = build_commands(source)
    module = _load_verified_launcher()
    _configure_launcher(module, source, commands)
    freshness = {
        "local_exists": module.OUTPUT.exists(),
        "remote_exists": module.remote_path_exists(module.REMOTE),
    }
    if freshness["local_exists"] or freshness["remote_exists"]:
        raise ValueError(
            f"attempt tag is not fresh: {freshness}"
        )
    try:
        return module.main()
    finally:
        if module.OUTPUT.exists():
            shutil.copy2(
                Path(__file__).resolve(),
                module.OUTPUT / Path(__file__).name,
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=(*SOURCES, "all"),
        required=True,
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--check-freshness", action="store_true")
    action.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    names = tuple(SOURCES) if args.source == "all" else (args.source,)
    if args.dry_run:
        payload = {
            name: build_dry_run_payload(SOURCES[name])
            for name in names
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.check_freshness:
        payload = {
            name: check_freshness(SOURCES[name])
            for name in names
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if all(row["fresh"] for row in payload.values()) else 2
    if len(names) != 1:
        raise ValueError("--run requires one explicit source")
    return run_source(SOURCES[names[0]])


if __name__ == "__main__":
    raise SystemExit(main())

