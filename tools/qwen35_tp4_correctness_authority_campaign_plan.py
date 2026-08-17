from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


SCHEMA_VERSION = "qwen35.tp4-correctness-authority-campaign-plan.v1"
PLAN_NAME = "campaign_plan.json"
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
_SAFE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class CampaignChild:
    name: str
    plan_path: Path
    authority_dir: Path
    authorization_path: Path
    consumed_authorization_path: Path
    receipt_path: Path
    failure_path: Path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _regular_file(path, label):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file")
    return path.resolve()


def _absolute(path, label):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return path.resolve()


def _safe_tag(value):
    if not isinstance(value, str) or not _SAFE_TAG.fullmatch(value):
        raise ValueError("campaign tag is invalid")
    return value


def _load_json(path):
    path = _regular_file(path, "campaign plan")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("campaign plan is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError("campaign plan is invalid")
    return payload


def _atomic_write(path, payload):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_identity(name, plan):
    field = (
        "frozen_source_tree_sha256"
        if name == "tp4_root_logit"
        else "source_tree_sha256"
    )
    value = plan.get(field)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} source identity is invalid")
    if name == "tp4_root_logit" and value != TP4_ROOT_SOURCE_TREE_SHA256:
        raise ValueError("tp4_root_logit source identity mismatch")
    return value


def _child_payload(child, verifier):
    if not isinstance(child, CampaignChild):
        raise ValueError("campaign child inventory is invalid")
    if not callable(verifier):
        raise ValueError(f"{child.name} plan verifier is required")
    plan_path = _regular_file(child.plan_path, f"{child.name} plan")
    plan = verifier(plan_path)
    if not isinstance(plan, dict):
        raise ValueError(f"{child.name} plan is invalid")
    if plan.get("ssh_target") != SSH_TARGET:
        raise ValueError(f"{child.name} SSH target mismatch")
    run_tag = plan.get("run_tag")
    if not isinstance(run_tag, str) or not run_tag:
        raise ValueError(f"{child.name} run tag is invalid")
    if plan.get("model_manifest_sha256") != MODEL_MANIFEST_SHA256:
        raise ValueError(f"{child.name} model identity mismatch")
    paths = {
        "plan_path": plan_path,
        "authority_dir": _absolute(
            child.authority_dir,
            f"{child.name} authority directory",
        ),
        "authorization_path": _absolute(
            child.authorization_path,
            f"{child.name} authorization",
        ),
        "consumed_authorization_path": _absolute(
            child.consumed_authorization_path,
            f"{child.name} consumed authorization",
        ),
        "receipt_path": _absolute(
            child.receipt_path,
            f"{child.name} receipt",
        ),
        "failure_path": _absolute(
            child.failure_path,
            f"{child.name} failure",
        ),
    }
    if any(
        paths[field].exists()
        for field in (
            "consumed_authorization_path",
            "receipt_path",
            "failure_path",
        )
    ):
        raise ValueError("campaign output target exists")
    if not paths["authorization_path"].is_file():
        raise ValueError(f"{child.name} authorization is missing")
    payload = {
        "name": child.name,
        "run_tag": run_tag,
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "source_tree_sha256": _source_identity(child.name, plan),
        "model_manifest_sha256": MODEL_MANIFEST_SHA256,
        "authority_dir": str(paths["authority_dir"]),
        "authorization_path": str(paths["authorization_path"]),
        "consumed_authorization_path": str(
            paths["consumed_authorization_path"]
        ),
        "receipt_path": str(paths["receipt_path"]),
        "failure_path": str(paths["failure_path"]),
    }
    if plan.get("resource_policy") is not None:
        if (
            plan["resource_policy"] != "controlled_shared"
            or not isinstance(
                plan.get("resource_baseline_sha256"),
                str,
            )
            or len(plan["resource_baseline_sha256"]) != 64
        ):
            raise ValueError(f"{child.name} resource identity mismatch")
        payload.update({
            "resource_policy": plan["resource_policy"],
            "resource_baseline_sha256": plan[
                "resource_baseline_sha256"
            ],
        })
    return payload


def _payload(
    *,
    repo_root,
    output_dir,
    campaign_tag,
    children,
    child_plan_verifiers,
    adapter_output_dir,
    bundle_output_dir,
):
    if (
        not isinstance(children, (tuple, list))
        or [child.name for child in children] != list(CHILD_ORDER)
        or set(child_plan_verifiers) != set(CHILD_ORDER)
    ):
        raise ValueError("campaign child inventory is invalid")
    child_rows = [
        _child_payload(child, child_plan_verifiers[child.name])
        for child in children
    ]
    adapter_output_dir = _absolute(
        adapter_output_dir,
        "adapter output directory",
    )
    bundle_output_dir = _absolute(
        bundle_output_dir,
        "bundle output directory",
    )
    if (
        adapter_output_dir == bundle_output_dir
        or adapter_output_dir.exists()
        or bundle_output_dir.exists()
    ):
        raise ValueError("campaign output target exists")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign_tag": _safe_tag(campaign_tag),
        "repo_root": str(_absolute(repo_root, "repository root")),
        "plan_output_dir": str(
            _absolute(output_dir, "plan output directory")
        ),
        "ssh_target": SSH_TARGET,
        "execution_env": dict(EXECUTION_ENV),
        "child_order": list(CHILD_ORDER),
        "stage_order": list(STAGE_ORDER),
        "children": child_rows,
        "adapter_output_dir": str(adapter_output_dir),
        "bundle_output_dir": str(bundle_output_dir),
        "prerequisite_path": str(
            bundle_output_dir / "correctness_prerequisites.json"
        ),
        "benchmark_execution_authorized": False,
        "execution_performed": False,
        "claim_boundary": (
            "correctness authority campaign only; no benchmark, "
            "performance, cache, memory, compression, or quality claim"
        ),
    }
    policies = {
        row.get("resource_policy") for row in child_rows
    }
    if policies != {None}:
        baseline_shas = {
            row.get("resource_baseline_sha256")
            for row in child_rows
        }
        if policies != {"controlled_shared"} or len(baseline_shas) != 1:
            raise ValueError("campaign child resource identity mismatch")
        payload.update({
            "resource_policy": "controlled_shared",
            "resource_baseline_sha256": baseline_shas.pop(),
        })
    return payload


def build_campaign_plan(
    *,
    repo_root,
    output_dir,
    campaign_tag,
    children,
    child_plan_verifiers,
    adapter_output_dir,
    bundle_output_dir,
):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("campaign output target exists")
    payload = _payload(
        repo_root=repo_root,
        output_dir=output_dir,
        campaign_tag=campaign_tag,
        children=children,
        child_plan_verifiers=child_plan_verifiers,
        adapter_output_dir=adapter_output_dir,
        bundle_output_dir=bundle_output_dir,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    try:
        _atomic_write(output_dir / PLAN_NAME, payload)
        return verify_campaign_plan(
            output_dir / PLAN_NAME,
            child_plan_verifiers=child_plan_verifiers,
        )
    except BaseException:
        if output_dir.exists():
            for path in output_dir.iterdir():
                path.unlink()
            output_dir.rmdir()
        raise


def verify_campaign_plan(path, *, child_plan_verifiers):
    path = Path(path).resolve()
    payload = _load_json(path)
    required = {
        "schema_version",
        "campaign_tag",
        "repo_root",
        "plan_output_dir",
        "ssh_target",
        "execution_env",
        "child_order",
        "stage_order",
        "children",
        "adapter_output_dir",
        "bundle_output_dir",
        "prerequisite_path",
        "benchmark_execution_authorized",
        "execution_performed",
        "claim_boundary",
    }
    resource_policy = payload.get("resource_policy")
    if resource_policy is not None:
        required.update({
            "resource_policy",
            "resource_baseline_sha256",
        })
    if (
        set(payload) != required
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("plan_output_dir") != str(path.parent)
        or payload.get("ssh_target") != SSH_TARGET
        or payload.get("execution_env") != EXECUTION_ENV
        or payload.get("child_order") != list(CHILD_ORDER)
        or payload.get("stage_order") != list(STAGE_ORDER)
        or payload.get("benchmark_execution_authorized") is not False
        or payload.get("execution_performed") is not False
        or not isinstance(payload.get("claim_boundary"), str)
    ):
        raise ValueError("campaign plan schema mismatch")
    _safe_tag(payload["campaign_tag"])
    if set(child_plan_verifiers) != set(CHILD_ORDER):
        raise ValueError("campaign child inventory is invalid")
    rows = payload.get("children")
    if (
        not isinstance(rows, list)
        or [row.get("name") for row in rows] != list(CHILD_ORDER)
    ):
        raise ValueError("campaign child inventory is invalid")
    for row in rows:
        expected_row = {
            "name",
            "run_tag",
            "plan_path",
            "plan_sha256",
            "source_tree_sha256",
            "model_manifest_sha256",
            "authority_dir",
            "authorization_path",
            "consumed_authorization_path",
            "receipt_path",
            "failure_path",
        }
        if resource_policy is not None:
            expected_row.update({
                "resource_policy",
                "resource_baseline_sha256",
            })
        if not isinstance(row, dict) or set(row) != expected_row:
            raise ValueError("campaign child schema mismatch")
        plan_path = _regular_file(row["plan_path"], f"{row['name']} plan")
        if _sha256(plan_path) != row["plan_sha256"]:
            raise ValueError(f"{row['name']} plan SHA mismatch")
        plan = child_plan_verifiers[row["name"]](plan_path)
        if (
            plan.get("run_tag") != row["run_tag"]
            or plan.get("ssh_target") != SSH_TARGET
            or _source_identity(row["name"], plan)
            != row["source_tree_sha256"]
            or plan.get("model_manifest_sha256")
            != row["model_manifest_sha256"]
            or (
                resource_policy is not None
                and (
                    resource_policy != "controlled_shared"
                    or plan.get("resource_policy")
                    != resource_policy
                    or row.get("resource_policy")
                    != resource_policy
                    or plan.get("resource_baseline_sha256")
                    != payload["resource_baseline_sha256"]
                    or row.get("resource_baseline_sha256")
                    != payload["resource_baseline_sha256"]
                )
            )
        ):
            raise ValueError(f"{row['name']} plan identity mismatch")
        for field in (
            "authority_dir",
            "authorization_path",
            "consumed_authorization_path",
            "receipt_path",
            "failure_path",
        ):
            if not Path(row[field]).is_absolute():
                raise ValueError("campaign child path is invalid")
    if (
        payload["prerequisite_path"]
        != str(
            Path(payload["bundle_output_dir"])
            / "correctness_prerequisites.json"
        )
    ):
        raise ValueError("campaign prerequisite path mismatch")
    return payload
