from __future__ import annotations

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


campaign = _load(
    "qwen35_tp4_correctness_authority_campaign_plan",
    "qwen35_tp4_correctness_authority_campaign_plan.py",
)


def test_campaign_freezes_current_root_source_identity():
    assert campaign.TP4_ROOT_SOURCE_TREE_SHA256 == (
        "ec19a8fa68abfba72e9594bdd1e05428"
        "b0add9169d3dbdde24190686c013411f"
    )


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _expect_value_error(function, fragment):
    try:
        function()
    except ValueError as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(
            f"expected ValueError containing {fragment!r}"
        )


def _fixture(root, *, controlled_shared=False):
    names = campaign.CHILD_ORDER
    children = []
    verifiers = {}
    for index, name in enumerate(names):
        child_root = root / name
        plan_path = child_root / "remote_execution_plan.json"
        authorization_path = child_root / "authorization.json"
        authority_dir = root / "authorities" / name
        source = (
            campaign.TP4_ROOT_SOURCE_TREE_SHA256
            if name == "tp4_root_logit"
            else str(index + 1) * 64
        )
        payload = {
            "run_tag": f"{name}-run",
            "ssh_target": campaign.SSH_TARGET,
            "model_manifest_sha256": campaign.MODEL_MANIFEST_SHA256,
            "authority_dir": str(authority_dir),
        }
        if controlled_shared:
            payload.update({
                "resource_policy": "controlled_shared",
                "resource_baseline_sha256": "f" * 64,
            })
        payload[
            "frozen_source_tree_sha256"
            if name == "tp4_root_logit"
            else "source_tree_sha256"
        ] = source
        _write_json(plan_path, payload)
        _write_json(authorization_path, {"consumed": False})
        children.append(campaign.CampaignChild(
            name=name,
            plan_path=plan_path,
            authority_dir=authority_dir,
            authorization_path=authorization_path,
            consumed_authorization_path=(
                child_root / "consumed_authorization.json"
            ),
            receipt_path=child_root / "execution_receipt.json",
            failure_path=child_root / "execution_failure.json",
        ))
        verifiers[name] = (
            lambda path, expected=payload: dict(expected)
        )
    return tuple(children), verifiers


def test_builder_freezes_exact_campaign_and_child_identities():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        children, verifiers = _fixture(root)
        output = root / "campaign-plan"
        plan = campaign.build_campaign_plan(
            repo_root=ROOT,
            output_dir=output,
            campaign_tag="correctness-r1",
            children=children,
            child_plan_verifiers=verifiers,
            adapter_output_dir=root / "adapter",
            bundle_output_dir=root / "bundle",
        )

        assert plan["child_order"] == list(campaign.CHILD_ORDER)
        assert plan["stage_order"] == list(campaign.STAGE_ORDER)
        assert plan["ssh_target"] == campaign.SSH_TARGET
        assert plan["execution_env"] == campaign.EXECUTION_ENV
        assert plan["benchmark_execution_authorized"] is False
        assert [row["name"] for row in plan["children"]] == list(
            campaign.CHILD_ORDER
        )
        assert all(len(row["plan_sha256"]) == 64 for row in plan["children"])
        assert campaign.verify_campaign_plan(
            output / campaign.PLAN_NAME,
            child_plan_verifiers=verifiers,
        ) == plan


def test_builder_binds_one_controlled_shared_resource_identity():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        children, verifiers = _fixture(
            root,
            controlled_shared=True,
        )
        plan = campaign.build_campaign_plan(
            repo_root=ROOT,
            output_dir=root / "campaign-plan",
            campaign_tag="correctness-shared-r1",
            children=children,
            child_plan_verifiers=verifiers,
            adapter_output_dir=root / "adapter",
            bundle_output_dir=root / "bundle",
        )
        assert plan["resource_policy"] == "controlled_shared"
        assert plan["resource_baseline_sha256"] == "f" * 64
        assert all(
            row["resource_policy"] == "controlled_shared"
            and row["resource_baseline_sha256"] == "f" * 64
            for row in plan["children"]
        )
        assert plan["benchmark_execution_authorized"] is False


def test_builder_rejects_child_identity_and_inventory_drift():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        children, verifiers = _fixture(root)
        wrong_target = dict(verifiers)
        wrong_target["cached_continuation"] = lambda path: {
            **json.loads(Path(path).read_text()),
            "ssh_target": "wrong@example.com",
        }
        cases = (
            (tuple(reversed(children)), verifiers, "child inventory"),
            (children, wrong_target, "SSH target"),
        )
        for index, (rows, child_verifiers, fragment) in enumerate(cases):
            _expect_value_error(
                lambda rows=rows, child_verifiers=child_verifiers: (
                    campaign.build_campaign_plan(
                        repo_root=ROOT,
                        output_dir=root / f"plan-{index}",
                        campaign_tag=f"correctness-r{index + 2}",
                        children=rows,
                        child_plan_verifiers=child_verifiers,
                        adapter_output_dir=root / f"adapter-{index}",
                        bundle_output_dir=root / f"bundle-{index}",
                    )
                ),
                fragment,
            )


def test_builder_rejects_existing_outputs_and_unsafe_campaign_tag():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        children, verifiers = _fixture(root)
        adapter_output = root / "adapter"
        adapter_output.mkdir()
        _expect_value_error(
            lambda: campaign.build_campaign_plan(
                repo_root=ROOT,
                output_dir=root / "plan",
                campaign_tag="correctness-r1",
                children=children,
                child_plan_verifiers=verifiers,
                adapter_output_dir=adapter_output,
                bundle_output_dir=root / "bundle",
            ),
            "output target exists",
        )
        _expect_value_error(
            lambda: campaign.build_campaign_plan(
                repo_root=ROOT,
                output_dir=root / "plan-unsafe",
                campaign_tag="../unsafe",
                children=children,
                child_plan_verifiers=verifiers,
                adapter_output_dir=root / "adapter-safe",
                bundle_output_dir=root / "bundle-safe",
            ),
            "campaign tag",
        )


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 correctness authority campaign plan tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
