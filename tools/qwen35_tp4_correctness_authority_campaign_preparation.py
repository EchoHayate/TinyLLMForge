from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


SCHEMA_VERSION = (
    "qwen35.tp4-correctness-campaign-preparation.v1"
)
MANIFEST_NAME = "preparation_manifest.json"
SSH_TARGET = "sitian@10.232.195.203"
EXECUTION_ENV = {
    "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian",
}
MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38f"
    "e69982efedaf4a3e631ad0b14aad7dd0"
)
TP4_ROOT_SOURCE_TREE_SHA256 = (
    "ec19a8fa68abfba72e9594bdd1e05428"
    "b0add9169d3dbdde24190686c013411f"
)
CHILD_ORDER = (
    "tp4_root_logit",
    "cached_continuation",
    "engine_correctness",
)
STAGE_ORDER = (
    "root_logit",
    "cached_continuation",
    "engine_correctness",
    "adapt_authorities",
    "build_bundle",
    "verify_bundle",
)
RESOURCE_BASELINE_NAME = "resource_baseline.json"


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


def _default_dependencies():
    _load_module(
        "qwen35_tp4_hybrid_prefix_benchmark_contract",
        "qwen35_tp4_hybrid_prefix_benchmark_contract.py",
    )
    engine_authorization = _load_module(
        "qwen35_tp4_engine_remote_execution_authorization_for_preparation",
        "qwen35_tp4_engine_remote_execution_authorization.py",
    )
    return {
        "root_plan": _load_module(
            "qwen35_tp4_root_logit_remote_execution_plan_for_preparation",
            "qwen35_tp4_root_logit_remote_execution_plan.py",
        ),
        "root_authorization": _load_module(
            "qwen35_tp4_root_logit_remote_execution_authorization_for_preparation",
            "qwen35_tp4_root_logit_remote_execution_authorization.py",
        ),
        "cached_plan": _load_module(
            "qwen35_tp4_cached_continuation_remote_execution_plan_for_preparation",
            "qwen35_tp4_cached_continuation_remote_execution_plan.py",
        ),
        "cached_authorization": engine_authorization,
        "engine_plan": _load_module(
            "qwen35_tp4_engine_remote_execution_plan_for_preparation",
            "qwen35_tp4_engine_remote_execution_plan.py",
        ),
        "engine_authorization": engine_authorization,
        "campaign_plan": _load_module(
            "qwen35_tp4_correctness_authority_campaign_plan_for_preparation",
            "qwen35_tp4_correctness_authority_campaign_plan.py",
        ),
        "campaign_authorization": _load_module(
            "qwen35_tp4_correctness_authority_campaign_authorization_for_preparation",
            "qwen35_tp4_correctness_authority_campaign_authorization.py",
        ),
    }


def _canonical_bytes(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _regular_file(path, label):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file")
    return path.resolve()


def _under(root, path, label):
    root = Path(root).resolve()
    path = Path(path)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes preparation root") from error
    return path


def _atomic_write(path, payload):
    path = Path(path)
    if path.exists():
        raise ValueError("preparation manifest already exists")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(_canonical_bytes(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path, label):
    path = _regular_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is invalid")
    return payload


def _plan_name(module, fallback):
    value = getattr(module, "PLAN_NAME", fallback)
    if not isinstance(value, str) or not value:
        raise ValueError("plan module name is invalid")
    return value


def _source_identity(name, plan):
    value = plan.get(
        "frozen_source_tree_sha256"
        if name == "tp4_root_logit"
        else "source_tree_sha256"
    )
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} source identity is invalid")
    return value


def _model_identity(plan):
    value = plan.get("model_manifest_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("child model identity is invalid")
    return value


def _configuration_identity(configuration, inventory):
    required = {
        "source_tree_sha256",
        "model_manifest_sha256",
        "workload_manifest_sha256",
        "gpu_indices",
        "dist_port",
        "master_port",
    }
    if (
        not isinstance(configuration, dict)
        or not required.issubset(configuration)
        or not isinstance(inventory, dict)
        or inventory.get("source_tree_sha256")
        != configuration.get("source_tree_sha256")
    ):
        raise ValueError("preparation configuration identity mismatch")
    return {
        "source_tree_sha256": configuration["source_tree_sha256"],
        "model_manifest_sha256": configuration[
            "model_manifest_sha256"
        ],
        "workload_manifest_sha256": configuration[
            "workload_manifest_sha256"
        ],
        "gpu_indices": configuration["gpu_indices"],
        "ports": {
            "dist_port": configuration["dist_port"],
            "master_port": configuration["master_port"],
        },
    }


def _child_layout(root, name):
    child_root = root / "children" / name
    runtime = root / "runtime" / name
    return {
        "plan_dir": child_root / "plan",
        "authorization_path": child_root / "authorization.json",
        "consumed_authorization_path": (
            runtime / "consumed_authorization.json"
        ),
        "receipt_path": runtime / "execution_receipt.json",
        "failure_path": runtime / "execution_failure.json",
    }


def _authority_dir(name, plan, plan_path, module):
    if name == "tp4_root_logit":
        try:
            value = plan["stage_inputs"]["verify"][
                "local_artifact_dir"
            ]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "root child authority directory is not plan-bound"
            ) from error
        path = Path(value)
    else:
        downloaded_name = getattr(
            module,
            "DOWNLOADED_AUTHORITY_NAME",
            None,
        )
        if not isinstance(downloaded_name, str) or not downloaded_name:
            raise ValueError(
                f"{name} downloaded authority name is invalid"
            )
        path = Path(plan_path).parent / downloaded_name
        try:
            argv = plan["commands"]["local_verify"]["argv"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"{name} local verifier is not plan-bound"
            ) from error
        authority_root = path.resolve()
        bound_paths = []
        if isinstance(argv, list):
            for value in argv:
                if (
                    not isinstance(value, (str, os.PathLike))
                    or not Path(value).is_absolute()
                ):
                    continue
                candidate = Path(value).resolve()
                try:
                    candidate.relative_to(authority_root)
                except ValueError:
                    continue
                bound_paths.append(candidate)
        if not bound_paths:
            raise ValueError(
                f"{name} authority directory is not plan-bound"
            )
    if not path.is_absolute():
        raise ValueError(f"{name} authority directory is invalid")
    return path.resolve()


def _campaign_layout(root):
    campaign_root = root / "campaign"
    runtime = root / "runtime" / "campaign"
    return {
        "plan_dir": campaign_root / "plan",
        "authorization_path": campaign_root / "authorization.json",
        "consumed_authorization_path": (
            runtime / "consumed_authorization.json"
        ),
        "receipt_path": runtime / "campaign_receipt.json",
        "failure_path": runtime / "campaign_failure.json",
        "adapter_output_dir": root / "adapter",
        "bundle_output_dir": root / "bundle",
    }


def _child_verifiers(dependencies):
    return {
        "tp4_root_logit": dependencies[
            "root_plan"
        ].verify_remote_execution_plan,
        "cached_continuation": dependencies[
            "cached_plan"
        ].verify_remote_execution_plan,
        "engine_correctness": dependencies[
            "engine_plan"
        ].verify_remote_execution_plan,
    }


def _manifest_child(
    *,
    name,
    plan,
    plan_path,
    authorization_path,
    layout,
    authority_dir,
):
    payload = {
        "name": name,
        "run_tag": plan["run_tag"],
        "plan_path": str(Path(plan_path).resolve()),
        "plan_sha256": _sha256(plan_path),
        "authorization_path": str(
            Path(authorization_path).resolve()
        ),
        "authorization_sha256": _sha256(authorization_path),
        "consumed_authorization_path": str(
            layout["consumed_authorization_path"].resolve()
        ),
        "receipt_path": str(layout["receipt_path"].resolve()),
        "failure_path": str(layout["failure_path"].resolve()),
        "authority_dir": str(Path(authority_dir).resolve()),
        "source_tree_sha256": _source_identity(name, plan),
        "model_manifest_sha256": _model_identity(plan),
    }
    if plan.get("resource_policy") is not None:
        payload.update({
            "resource_policy": plan["resource_policy"],
            "resource_baseline_sha256": plan[
                "resource_baseline_sha256"
            ],
        })
    return payload


def _manifest_payload(
    *,
    root,
    campaign_tag,
    child_rows,
    inputs,
    campaign_plan,
    campaign_plan_path,
    campaign_authorization_path,
    campaign_layout,
):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "classification": "READY",
        "preparation_root": str(root),
        "campaign_tag": campaign_tag,
        "ssh_target": SSH_TARGET,
        "execution_env": dict(EXECUTION_ENV),
        "child_order": list(CHILD_ORDER),
        "stage_order": list(STAGE_ORDER),
        "inputs": inputs,
        "children": child_rows,
        "campaign": {
            "plan_path": str(Path(campaign_plan_path).resolve()),
            "plan_sha256": _sha256(campaign_plan_path),
            "authorization_path": str(
                Path(campaign_authorization_path).resolve()
            ),
            "authorization_sha256": _sha256(
                campaign_authorization_path
            ),
            "consumed_authorization_path": str(
                campaign_layout[
                    "consumed_authorization_path"
                ].resolve()
            ),
            "receipt_path": str(
                campaign_layout["receipt_path"].resolve()
            ),
            "failure_path": str(
                campaign_layout["failure_path"].resolve()
            ),
        },
        "adapter_output_dir": campaign_plan["adapter_output_dir"],
        "bundle_output_dir": campaign_plan["bundle_output_dir"],
        "model_manifest_sha256": MODEL_MANIFEST_SHA256,
        "execution_performed": False,
        "benchmark_execution_authorized": False,
        "claim_boundary": (
            "preparation only; no remote execution, correctness, "
            "performance, cache, memory, compression, or quality claim"
        ),
    }
    policies = {
        row.get("resource_policy") for row in child_rows
    }
    if policies != {None}:
        if len(policies) != 1:
            raise ValueError("child resource policy mismatch")
        baseline_shas = {
            row.get("resource_baseline_sha256")
            for row in child_rows
        }
        if len(baseline_shas) != 1:
            raise ValueError("child resource baseline mismatch")
        payload.update({
            "resource_policy": policies.pop(),
            "resource_baseline_path": inputs[
                "resource_baseline_path"
            ],
            "resource_baseline_sha256": baseline_shas.pop(),
        })
    return payload


def _validate_payload(payload, root, dependencies):
    required = {
        "schema_version",
        "classification",
        "preparation_root",
        "campaign_tag",
        "ssh_target",
        "execution_env",
        "child_order",
        "stage_order",
        "inputs",
        "children",
        "campaign",
        "adapter_output_dir",
        "bundle_output_dir",
        "model_manifest_sha256",
        "execution_performed",
        "benchmark_execution_authorized",
        "claim_boundary",
    }
    resource_policy = payload.get("resource_policy")
    if resource_policy is not None:
        required.update({
            "resource_policy",
            "resource_baseline_path",
            "resource_baseline_sha256",
        })
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["classification"] != "READY"
        or payload["preparation_root"] != str(root)
        or payload["ssh_target"] != SSH_TARGET
        or payload["execution_env"] != EXECUTION_ENV
        or payload["child_order"] != list(CHILD_ORDER)
        or payload["stage_order"] != list(STAGE_ORDER)
        or payload["model_manifest_sha256"]
        != MODEL_MANIFEST_SHA256
        or payload["execution_performed"] is not False
        or payload["benchmark_execution_authorized"] is not False
        or not isinstance(payload["claim_boundary"], str)
    ):
        raise ValueError("preparation manifest schema mismatch")

    rows = payload["children"]
    inputs = payload["inputs"]
    expected_inputs = {
        "configuration_path",
        "configuration_sha256",
        "source_inventory_path",
        "source_inventory_sha256",
    }
    if resource_policy is not None:
        expected_inputs.update({
            "resource_baseline_path",
            "resource_baseline_sha256",
        })
    if not isinstance(inputs, dict) or set(inputs) != expected_inputs:
        raise ValueError("preparation input schema mismatch")
    configuration_path = _under(
        root,
        inputs["configuration_path"],
        "preparation configuration",
    )
    source_inventory_path = _under(
        root,
        inputs["source_inventory_path"],
        "preparation source inventory",
    )
    if _sha256(
        _regular_file(
            configuration_path,
            "preparation configuration",
        )
    ) != inputs["configuration_sha256"]:
        raise ValueError("preparation configuration SHA mismatch")
    if _sha256(
        _regular_file(
            source_inventory_path,
            "preparation source inventory",
        )
    ) != inputs["source_inventory_sha256"]:
        raise ValueError("preparation source inventory SHA mismatch")
    if resource_policy is not None:
        resource_baseline_path = _under(
            root,
            inputs["resource_baseline_path"],
            "preparation resource baseline",
        )
        if (
            _sha256(_regular_file(
                resource_baseline_path,
                "preparation resource baseline",
            ))
            != inputs["resource_baseline_sha256"]
            or payload["resource_baseline_path"]
            != str(resource_baseline_path)
            or payload["resource_baseline_sha256"]
            != inputs["resource_baseline_sha256"]
        ):
            raise ValueError(
                "preparation resource baseline SHA mismatch"
            )
    configuration = _load_json(
        configuration_path,
        "preparation configuration",
    )
    source_inventory = _load_json(
        source_inventory_path,
        "preparation source inventory",
    )
    configuration_identity = _configuration_identity(
        configuration,
        source_inventory,
    )
    if (
        not isinstance(rows, list)
        or [row.get("name") for row in rows] != list(CHILD_ORDER)
    ):
        raise ValueError("preparation child inventory mismatch")
    verifiers = _child_verifiers(dependencies)
    plans = {}
    for row in rows:
        expected_row = {
            "name",
            "run_tag",
            "plan_path",
            "plan_sha256",
            "authorization_path",
            "authorization_sha256",
            "consumed_authorization_path",
            "receipt_path",
            "failure_path",
            "authority_dir",
            "source_tree_sha256",
            "model_manifest_sha256",
        }
        if resource_policy is not None:
            expected_row.update({
                "resource_policy",
                "resource_baseline_sha256",
            })
        if not isinstance(row, dict) or set(row) != expected_row:
            raise ValueError("preparation child schema mismatch")
        name = row["name"]
        plan_path = _under(root, row["plan_path"], "child plan")
        authorization_path = _under(
            root,
            row["authorization_path"],
            "child authorization",
        )
        if _sha256(_regular_file(plan_path, "child plan")) != row[
            "plan_sha256"
        ]:
            raise ValueError("child plan SHA mismatch")
        if _sha256(
            _regular_file(authorization_path, "child authorization")
        ) != row["authorization_sha256"]:
            raise ValueError("child authorization SHA mismatch")
        plan = verifiers[name](plan_path)
        plans[name] = plan
        if (
            plan.get("run_tag") != row["run_tag"]
            or _source_identity(name, plan)
            != row["source_tree_sha256"]
            or _model_identity(plan) != row["model_manifest_sha256"]
            or row["model_manifest_sha256"]
            != MODEL_MANIFEST_SHA256
            or (
                resource_policy is not None
                and (
                    plan.get("resource_policy") != resource_policy
                    or row.get("resource_policy") != resource_policy
                    or plan.get("resource_baseline_sha256")
                    != payload["resource_baseline_sha256"]
                    or row.get("resource_baseline_sha256")
                    != payload["resource_baseline_sha256"]
                )
            )
        ):
            raise ValueError("child plan identity mismatch")
        if name != "tp4_root_logit":
            local_inputs = plan.get("local_inputs")
            if (
                not isinstance(local_inputs, dict)
                or local_inputs.get("source_inventory")
                != str(source_inventory_path)
                or plan.get("source_tree_sha256")
                != configuration_identity["source_tree_sha256"]
                or plan.get("model_manifest_sha256")
                != configuration_identity["model_manifest_sha256"]
                or local_inputs.get("workload_manifest_sha256")
                != configuration_identity[
                    "workload_manifest_sha256"
                ]
                or plan.get("gpu_indices")
                != configuration_identity["gpu_indices"]
                or plan.get("ports")
                != configuration_identity["ports"]
            ):
                raise ValueError(
                    "preparation configuration identity mismatch"
                )
        authorization_payload = _load_json(
            authorization_path,
            "child authorization",
        )
        dependencies[
            {
                "tp4_root_logit": "root_authorization",
                "cached_continuation": "cached_authorization",
                "engine_correctness": "engine_authorization",
            }[name]
        ].validate_authorization(plan, authorization_payload)
        authority_path = None
        for field in (
            "consumed_authorization_path",
            "receipt_path",
            "failure_path",
            "authority_dir",
        ):
            future = (
                Path(row[field]).resolve()
                if field == "authority_dir"
                else _under(root, row[field], f"child {field}")
            )
            if future.exists():
                raise ValueError("preparation future output exists")
            if field == "authority_dir":
                authority_path = future
        expected_authority = _authority_dir(
            name,
            plan,
            plan_path,
            dependencies[
                {
                    "tp4_root_logit": "root_plan",
                    "cached_continuation": "cached_plan",
                    "engine_correctness": "engine_plan",
                }[name]
            ],
        )
        if authority_path != expected_authority:
            raise ValueError("child authority directory binding mismatch")

    campaign = payload["campaign"]
    if not isinstance(campaign, dict) or set(campaign) != {
        "plan_path",
        "plan_sha256",
        "authorization_path",
        "authorization_sha256",
        "consumed_authorization_path",
        "receipt_path",
        "failure_path",
    }:
        raise ValueError("preparation campaign schema mismatch")
    campaign_plan_path = _under(
        root,
        campaign["plan_path"],
        "campaign plan",
    )
    campaign_authorization_path = _under(
        root,
        campaign["authorization_path"],
        "campaign authorization",
    )
    if _sha256(_regular_file(campaign_plan_path, "campaign plan")) != (
        campaign["plan_sha256"]
    ):
        raise ValueError("campaign plan SHA mismatch")
    if _sha256(
        _regular_file(
            campaign_authorization_path,
            "campaign authorization",
        )
    ) != campaign["authorization_sha256"]:
        raise ValueError("campaign authorization SHA mismatch")
    campaign_plan = dependencies[
        "campaign_plan"
    ].verify_campaign_plan(
        campaign_plan_path,
        child_plan_verifiers=verifiers,
    )
    campaign_authorization = _load_json(
        campaign_authorization_path,
        "campaign authorization",
    )
    dependencies[
        "campaign_authorization"
    ].validate_authorization(campaign_plan, campaign_authorization)
    if (
        campaign_plan.get("campaign_tag") != payload["campaign_tag"]
        or campaign_plan.get("children") is None
        or [
            row["plan_sha256"]
            for row in campaign_plan["children"]
        ]
        != [row["plan_sha256"] for row in rows]
        or campaign_plan.get("adapter_output_dir")
        != payload["adapter_output_dir"]
        or campaign_plan.get("bundle_output_dir")
        != payload["bundle_output_dir"]
        or (
            resource_policy is not None
            and (
                campaign_plan.get("resource_policy")
                != resource_policy
                or campaign_plan.get("resource_baseline_sha256")
                != payload["resource_baseline_sha256"]
            )
        )
    ):
        raise ValueError("campaign preparation binding mismatch")
    for field in (
        "consumed_authorization_path",
        "receipt_path",
        "failure_path",
    ):
        future = _under(root, campaign[field], f"campaign {field}")
        if future.exists():
            raise ValueError("preparation future output exists")
    for field in ("adapter_output_dir", "bundle_output_dir"):
        future = _under(root, payload[field], field)
        if future.exists():
            raise ValueError("preparation future output exists")
    return payload


def verify_preparation_bundle(path, *, dependencies=None):
    path = _regular_file(path, "preparation manifest")
    root = path.parent.resolve()
    if path.name != MANIFEST_NAME:
        raise ValueError("preparation manifest name mismatch")
    payload = _load_json(path, "preparation manifest")
    return _validate_payload(
        payload,
        root,
        dependencies or _default_dependencies(),
    )


def prepare_campaign_bundle(
    *,
    repo_root,
    output_dir,
    campaign_tag,
    root_run_tag,
    cached_run_tag,
    engine_run_tag,
    configuration_path,
    source_inventory_path,
    remote_model_dir,
    remote_model_manifest,
    root_authorization_nonce,
    cached_authorization_nonce,
    engine_authorization_nonce,
    campaign_authorization_nonce,
    resource_policy=None,
    resource_baseline_path=None,
    dependencies=None,
):
    root = Path(output_dir).resolve()
    if root.exists():
        raise ValueError("preparation output already exists")
    run_tags = (root_run_tag, cached_run_tag, engine_run_tag)
    nonces = (
        root_authorization_nonce,
        cached_authorization_nonce,
        engine_authorization_nonce,
        campaign_authorization_nonce,
    )
    if len(set(run_tags)) != len(run_tags):
        raise ValueError("child run tags must be pairwise distinct")
    if len(set(nonces)) != len(nonces):
        raise ValueError("authorization nonces must be pairwise distinct")
    modules = dependencies or _default_dependencies()
    repo_root = Path(repo_root).resolve()
    configuration_path = Path(configuration_path).resolve()
    source_inventory_path = Path(source_inventory_path).resolve()
    if resource_policy is None:
        if resource_baseline_path is not None:
            raise ValueError(
                "strict preparation does not accept a resource baseline"
            )
    elif resource_policy != "controlled_shared":
        raise ValueError("preparation resource policy is unsupported")
    elif resource_baseline_path is None:
        raise ValueError("preparation resource baseline is required")
    else:
        resource_baseline_path = _regular_file(
            resource_baseline_path,
            "preparation resource baseline",
        )
    layouts = {
        name: _child_layout(root, name)
        for name in CHILD_ORDER
    }
    campaign_layout = _campaign_layout(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    try:
        frozen_inputs = root / "inputs"
        frozen_inputs.mkdir()
        frozen_configuration = (
            frozen_inputs / "executor_configuration.json"
        )
        frozen_source_inventory = (
            frozen_inputs / "source_inventory.json"
        )
        shutil.copyfile(configuration_path, frozen_configuration)
        shutil.copyfile(
            source_inventory_path,
            frozen_source_inventory,
        )
        input_manifest = {
            "configuration_path": str(
                frozen_configuration.resolve()
            ),
            "configuration_sha256": _sha256(
                frozen_configuration
            ),
            "source_inventory_path": str(
                frozen_source_inventory.resolve()
            ),
            "source_inventory_sha256": _sha256(
                frozen_source_inventory
            ),
        }
        frozen_resource_baseline = None
        if resource_policy is not None:
            frozen_resource_baseline = (
                frozen_inputs / RESOURCE_BASELINE_NAME
            )
            shutil.copyfile(
                resource_baseline_path,
                frozen_resource_baseline,
            )
            input_manifest.update({
                "resource_baseline_path": str(
                    frozen_resource_baseline.resolve()
                ),
                "resource_baseline_sha256": _sha256(
                    frozen_resource_baseline
                ),
            })
        child_specs = (
            (
                "tp4_root_logit",
                "root",
                root_run_tag,
                root_authorization_nonce,
            ),
            (
                "cached_continuation",
                "cached",
                cached_run_tag,
                cached_authorization_nonce,
            ),
            (
                "engine_correctness",
                "engine",
                engine_run_tag,
                engine_authorization_nonce,
            ),
        )
        child_rows = []
        verified_plans = {}
        for name, kind, run_tag, nonce in child_specs:
            layout = layouts[name]
            if kind == "root":
                plan_kwargs = {
                    "repo_root": repo_root,
                    "output_dir": layout["plan_dir"],
                    "run_tag": run_tag,
                }
            else:
                plan_kwargs = {
                    "repo_root": repo_root,
                    "configuration_path": frozen_configuration,
                    "source_inventory_path": frozen_source_inventory,
                    "output_dir": layout["plan_dir"],
                    "run_tag": run_tag,
                    "remote_model_dir": remote_model_dir,
                    "remote_model_manifest": remote_model_manifest,
                }
            if resource_policy is not None:
                plan_kwargs.update({
                    "resource_policy": resource_policy,
                    "resource_baseline_path": (
                        frozen_resource_baseline
                    ),
                })
            plan = modules[
                f"{kind}_plan"
            ].build_remote_execution_plan(**plan_kwargs)
            plan_path = (
                layout["plan_dir"]
                / _plan_name(
                    modules[f"{kind}_plan"],
                    "remote_execution_plan.json",
                )
            )
            plan = modules[
                f"{kind}_plan"
            ].verify_remote_execution_plan(plan_path)
            verified_plans[name] = plan
            modules[
                f"{kind}_authorization"
            ].produce_authorization(
                plan=plan,
                output_path=layout["authorization_path"],
                nonce=nonce,
            )
            authority_dir = _authority_dir(
                name,
                plan,
                plan_path,
                modules[f"{kind}_plan"],
            )
            child_rows.append(_manifest_child(
                name=name,
                plan=plan,
                plan_path=plan_path,
                authorization_path=layout["authorization_path"],
                layout=layout,
                authority_dir=authority_dir,
            ))

        verifiers = _child_verifiers(modules)
        campaign_children = [
            modules["campaign_plan"].CampaignChild(
                name=name,
                plan_path=child_rows[index]["plan_path"],
                authority_dir=child_rows[index]["authority_dir"],
                authorization_path=layouts[name]["authorization_path"],
                consumed_authorization_path=layouts[name][
                    "consumed_authorization_path"
                ],
                receipt_path=layouts[name]["receipt_path"],
                failure_path=layouts[name]["failure_path"],
            )
            for index, name in enumerate(CHILD_ORDER)
        ]
        campaign_plan = modules[
            "campaign_plan"
        ].build_campaign_plan(
            repo_root=repo_root,
            output_dir=campaign_layout["plan_dir"],
            campaign_tag=campaign_tag,
            children=campaign_children,
            child_plan_verifiers=verifiers,
            adapter_output_dir=campaign_layout[
                "adapter_output_dir"
            ],
            bundle_output_dir=campaign_layout["bundle_output_dir"],
        )
        campaign_plan_path = (
            campaign_layout["plan_dir"]
            / _plan_name(
                modules["campaign_plan"],
                "campaign_plan.json",
            )
        )
        campaign_plan = modules[
            "campaign_plan"
        ].verify_campaign_plan(
            campaign_plan_path,
            child_plan_verifiers=verifiers,
        )
        modules[
            "campaign_authorization"
        ].produce_authorization(
            plan=campaign_plan,
            output_path=campaign_layout["authorization_path"],
            nonce=campaign_authorization_nonce,
        )
        payload = _manifest_payload(
            root=root,
            campaign_tag=campaign_tag,
            child_rows=child_rows,
            inputs=input_manifest,
            campaign_plan=campaign_plan,
            campaign_plan_path=campaign_plan_path,
            campaign_authorization_path=campaign_layout[
                "authorization_path"
            ],
            campaign_layout=campaign_layout,
        )
        _validate_payload(payload, root, modules)
        _atomic_write(root / MANIFEST_NAME, payload)
        return verify_preparation_bundle(
            root / MANIFEST_NAME,
            dependencies=modules,
        )
    except BaseException:
        if root.exists():
            shutil.rmtree(root)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--campaign-tag", required=True)
    parser.add_argument("--root-run-tag", required=True)
    parser.add_argument("--cached-run-tag", required=True)
    parser.add_argument("--engine-run-tag", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--source-inventory", required=True)
    parser.add_argument("--remote-model-dir", required=True)
    parser.add_argument("--remote-model-manifest", required=True)
    parser.add_argument("--root-authorization-nonce", required=True)
    parser.add_argument("--cached-authorization-nonce", required=True)
    parser.add_argument("--engine-authorization-nonce", required=True)
    parser.add_argument("--campaign-authorization-nonce", required=True)
    parser.add_argument(
        "--resource-policy",
        choices=("controlled_shared",),
    )
    parser.add_argument("--resource-baseline")
    args = parser.parse_args(argv)
    result = prepare_campaign_bundle(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        campaign_tag=args.campaign_tag,
        root_run_tag=args.root_run_tag,
        cached_run_tag=args.cached_run_tag,
        engine_run_tag=args.engine_run_tag,
        configuration_path=args.configuration,
        source_inventory_path=args.source_inventory,
        remote_model_dir=args.remote_model_dir,
        remote_model_manifest=args.remote_model_manifest,
        root_authorization_nonce=args.root_authorization_nonce,
        cached_authorization_nonce=args.cached_authorization_nonce,
        engine_authorization_nonce=args.engine_authorization_nonce,
        campaign_authorization_nonce=(
            args.campaign_authorization_nonce
        ),
        resource_policy=args.resource_policy,
        resource_baseline_path=args.resource_baseline,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
