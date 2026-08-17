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


authorization = _load(
    "qwen35_tp4_correctness_authority_campaign_authorization",
    "qwen35_tp4_correctness_authority_campaign_authorization.py",
)


def _plan(root):
    children = []
    for index, name in enumerate((
        "tp4_root_logit",
        "cached_continuation",
        "engine_correctness",
    )):
        children.append({
            "name": name,
            "plan_sha256": str(index + 1) * 64,
        })
    return {
        "campaign_tag": "campaign-r1",
        "ssh_target": "sitian@10.232.195.203",
        "execution_env": {
            "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian",
        },
        "child_order": [row["name"] for row in children],
        "children": children,
        "adapter_output_dir": str(root / "adapter"),
        "bundle_output_dir": str(root / "bundle"),
        "benchmark_execution_authorized": False,
    }


def _expect_value_error(function, fragment):
    try:
        function()
    except ValueError as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(
            f"expected ValueError containing {fragment!r}"
        )


def test_authorization_binds_campaign_plan_and_consumes_once():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = _plan(root)
        active = root / "authorization.json"
        consumed = root / "consumed_authorization.json"
        payload = authorization.produce_authorization(
            plan=plan,
            output_path=active,
            nonce="operator-r1",
        )

        assert payload["consumed"] is False
        assert payload["benchmark_execution_authorized"] is False
        assert payload["child_plan_sha256s"] == {
            row["name"]: row["plan_sha256"]
            for row in plan["children"]
        }
        record = authorization.consume_authorization(
            plan=plan,
            authorization_path=active,
            consumed_path=consumed,
        )
        assert record["consumed"] is True
        assert not active.exists()
        assert json.loads(consumed.read_text()) == record
        _expect_value_error(
            lambda: authorization.consume_authorization(
                plan=plan,
                authorization_path=active,
                consumed_path=consumed,
            ),
            "authorization",
        )


def test_authorization_binds_controlled_shared_baseline_identity():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = {
            **_plan(root),
            "resource_policy": "controlled_shared",
            "resource_baseline_sha256": "f" * 64,
        }
        payload = authorization.produce_authorization(
            plan=plan,
            output_path=root / "authorization.json",
            nonce="operator-shared-r1",
        )
        assert payload["resource_policy"] == "controlled_shared"
        assert payload["resource_baseline_sha256"] == "f" * 64
        changed = dict(payload)
        changed["resource_baseline_sha256"] = "e" * 64
        _expect_value_error(
            lambda: authorization.validate_authorization(
                plan,
                changed,
            ),
            "plan",
        )


def test_authorization_rejects_plan_drift_and_unsafe_nonce():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = _plan(root)
        _expect_value_error(
            lambda: authorization.produce_authorization(
                plan=plan,
                output_path=root / "unsafe.json",
                nonce="../unsafe",
            ),
            "nonce",
        )
        active = root / "authorization.json"
        authorization.produce_authorization(
            plan=plan,
            output_path=active,
            nonce="operator-r2",
        )
        drifted = {
            **plan,
            "bundle_output_dir": str(root / "other-bundle"),
        }
        _expect_value_error(
            lambda: authorization.consume_authorization(
                plan=drifted,
                authorization_path=active,
                consumed_path=root / "consumed.json",
            ),
            "plan",
        )


def test_authorization_consumes_into_production_runtime_directory():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = _plan(root)
        active = root / "campaign" / "authorization.json"
        consumed = root / "runtime" / "campaign" / "consumed.json"
        authorization.produce_authorization(
            plan=plan,
            output_path=active,
            nonce="operator-r3",
        )
        result = authorization.consume_authorization(
            plan=plan,
            authorization_path=active,
            consumed_path=consumed,
        )
        assert result["consumed"] is True
        assert not active.exists()
        assert json.loads(consumed.read_text()) == result


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 correctness authority campaign authorization tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
