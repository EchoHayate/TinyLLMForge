from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys


def _load_module(module_name, filename):
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class CampaignCallbacks:
    child_executors: dict
    child_receipt_verifiers: dict
    adapt_callback: object
    build_callback: object
    prerequisite_validator: object


def _default_dependencies():
    contract = _load_module(
        "qwen35_tp4_hybrid_prefix_benchmark_contract",
        "qwen35_tp4_hybrid_prefix_benchmark_contract.py",
    )
    return {
        "root_plan": _load_module(
            "qwen35_tp4_root_logit_remote_execution_plan_for_campaign",
            "qwen35_tp4_root_logit_remote_execution_plan.py",
        ),
        "root_executor": _load_module(
            "qwen35_tp4_root_logit_remote_execution_executor_for_campaign",
            "qwen35_tp4_root_logit_remote_execution_executor.py",
        ),
        "root_receipt": _load_module(
            "qwen35_tp4_root_logit_remote_execution_receipt_for_campaign",
            "qwen35_tp4_root_logit_remote_execution_receipt.py",
        ),
        "cached_plan": _load_module(
            "qwen35_tp4_cached_continuation_remote_execution_plan_for_campaign",
            "qwen35_tp4_cached_continuation_remote_execution_plan.py",
        ),
        "cached_executor": _load_module(
            "qwen35_tp4_cached_continuation_remote_execution_executor_for_campaign",
            "qwen35_tp4_cached_continuation_remote_execution_executor.py",
        ),
        "cached_receipt": _load_module(
            "qwen35_tp4_cached_continuation_remote_execution_receipt_for_campaign",
            "qwen35_tp4_cached_continuation_remote_execution_receipt.py",
        ),
        "engine_plan": _load_module(
            "qwen35_tp4_engine_remote_execution_plan_for_campaign",
            "qwen35_tp4_engine_remote_execution_plan.py",
        ),
        "engine_executor": _load_module(
            "qwen35_tp4_engine_remote_execution_executor_for_campaign",
            "qwen35_tp4_engine_remote_execution_executor.py",
        ),
        "engine_receipt": _load_module(
            "qwen35_tp4_engine_remote_execution_receipt_for_campaign",
            "qwen35_tp4_engine_remote_execution_receipt.py",
        ),
        "adapter": _load_module(
            "qwen35_tp4_real_prerequisite_authority_adapter_for_campaign",
            "qwen35_tp4_real_prerequisite_authority_adapter.py",
        ),
        "builder": _load_module(
            "build_qwen35_tp4_performance_prerequisites_for_campaign",
            "build_qwen35_tp4_performance_prerequisites.py",
        ),
        "contract": contract,
    }


def build_campaign_callbacks(
    *,
    command_runner,
    root_stage_runner,
    root_verifier=None,
    dependencies=None,
):
    if not callable(command_runner):
        raise ValueError("explicit command runner is required")
    if not callable(root_stage_runner):
        raise ValueError("explicit root stage runner is required")
    modules = dependencies or _default_dependencies()

    def root_execute(*, child, execution_env):
        return modules["root_executor"].execute_verified_plan_file(
            plan_path=child["plan_path"],
            authorization_path=child["authorization_path"],
            consumed_authorization_path=child[
                "consumed_authorization_path"
            ],
            receipt_path=child["receipt_path"],
            failure_path=child["failure_path"],
            stage_runner=root_stage_runner,
            plan_verifier=modules[
                "root_plan"
            ].verify_remote_execution_plan,
            execution_env=execution_env,
            root_verifier=root_verifier,
        )

    def command_execute(kind, *, child, execution_env):
        return modules[f"{kind}_executor"].execute_verified_plan_file(
            plan_path=child["plan_path"],
            authorization_path=child["authorization_path"],
            consumed_authorization_path=child[
                "consumed_authorization_path"
            ],
            output_path=child["receipt_path"],
            failure_path=child["failure_path"],
            command_runner=command_runner,
            plan_verifier=modules[
                f"{kind}_plan"
            ].verify_remote_execution_plan,
            execution_env=execution_env,
        )

    def root_receipt_verify(**paths):
        return modules["root_receipt"].verify_receipt_files(
            **paths,
            plan_verifier=modules[
                "root_plan"
            ].verify_remote_execution_plan,
            root_verifier=root_verifier,
        )

    def adapt(*, runs, verification_output_dir):
        typed = tuple(
            modules["adapter"].RealAuthorityRun(
                name=row["name"],
                run_tag=row["run_tag"],
                authority_dir=Path(row["authority_dir"]),
                plan_path=Path(row["plan_path"]),
                consumed_authorization_path=Path(
                    row["consumed_authorization_path"]
                ),
                receipt_path=Path(row["receipt_path"]),
            )
            for row in runs
        )
        rows = modules["adapter"].adapt_real_authorities(
            runs=typed,
            verification_output_dir=verification_output_dir,
        )
        return [
            {
                field: (
                    str(getattr(row, field))
                    if field.endswith("_path")
                    else getattr(row, field)
                )
                for field in row.__dataclass_fields__
            }
            for row in rows
        ]

    def build(*, authorities, output_dir):
        typed = tuple(
            modules["builder"].AuthorityInput(
                **{
                    field: (
                        Path(row[field])
                        if field.endswith("_path")
                        else row[field]
                    )
                    for field in modules[
                        "builder"
                    ].AuthorityInput.__dataclass_fields__
                }
            )
            for row in authorities
        )
        result = modules["builder"].build_prerequisite_bundle(
            output_dir=output_dir,
            authorities=typed,
        )
        root = Path(output_dir)
        prerequisite = root / "correctness_prerequisites.json"
        return {
            "classification": result["classification"],
            "prerequisite_path": str(prerequisite),
            "prerequisite_sha256": result[
                "correctness_prerequisites_sha256"
            ],
            "owned_files": sorted(
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file() and not path.is_symlink()
            ),
        }

    def validate(path):
        status = modules["contract"].validate_prerequisites(path)
        return {
            "classification": status.classification,
            "authorized": status.authorized,
            "reasons": list(status.reasons),
        }

    return CampaignCallbacks(
        child_executors={
            "tp4_root_logit": root_execute,
            "cached_continuation": (
                lambda **kwargs: command_execute(
                    "cached",
                    **kwargs,
                )
            ),
            "engine_correctness": (
                lambda **kwargs: command_execute(
                    "engine",
                    **kwargs,
                )
            ),
        },
        child_receipt_verifiers={
            "tp4_root_logit": root_receipt_verify,
            "cached_continuation": (
                modules["cached_receipt"].verify_receipt_files
            ),
            "engine_correctness": (
                modules["engine_receipt"].verify_receipt_files
            ),
        },
        adapt_callback=adapt,
        build_callback=build,
        prerequisite_validator=validate,
    )
