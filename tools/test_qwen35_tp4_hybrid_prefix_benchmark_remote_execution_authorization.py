from __future__ import annotations

import copy
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
    "qwen35_tp4_hybrid_prefix_benchmark_remote_execution_authorization",
    (
        "qwen35_tp4_hybrid_prefix_benchmark_"
        "remote_execution_authorization.py"
    ),
)


def _plan():
    case_commands = []
    for index in range(70):
        case_commands.append({
            "case_id": f"case-{index:02d}",
            "dist_port": 22000 + index * 2,
            "master_port": 22001 + index * 2,
        })
    return {
        "schema_version": "qwen35.tp4-hybrid-prefix-benchmark.v1",
        "run_tag": "benchmark-authorized-r1",
        "worker_authorization": {
            "prerequisites_sha256": "a" * 64,
            "source_tree_sha256": "b" * 64,
            "model_manifest_sha256": "c" * 64,
            "workload_manifest_sha256": "d" * 64,
            "gpu_indices": [0, 1, 2, 3],
        },
        "case_commands": case_commands,
    }


def test_authorization_binds_plan_identities_and_all_case_ports():
    plan = _plan()
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "authorization.json"
        payload = authorization.produce_authorization(
            plan=plan,
            output_path=output,
            nonce="benchmark-once-r1",
        )

        assert json.loads(output.read_text(encoding="utf-8")) == payload
        assert payload == {
            "schema_version": (
                "qwen35.tp4-hybrid-prefix-benchmark-"
                "remote-execution-authorization.v1"
            ),
            "classification": "AUTHORIZED",
            "plan_sha256": authorization._canonical_sha(plan),
            "run_tag": "benchmark-authorized-r1",
            "prerequisites_sha256": "a" * 64,
            "source_tree_sha256": "b" * 64,
            "model_manifest_sha256": "c" * 64,
            "workload_manifest_sha256": "d" * 64,
            "gpu_indices": [0, 1, 2, 3],
            "case_port_pairs": [
                {
                    "case_id": f"case-{index:02d}",
                    "dist_port": 22000 + index * 2,
                    "master_port": 22001 + index * 2,
                }
                for index in range(70)
            ],
            "nonce": "benchmark-once-r1",
            "consumed": False,
        }
        assert authorization.validate_authorization(
            plan,
            payload,
        )["classification"] == "AUTHORIZED"


def test_authorization_rejects_identity_port_or_nonce_tamper():
    plan = _plan()
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "authorization.json"
        payload = authorization.produce_authorization(
            plan=plan,
            output_path=output,
            nonce="benchmark-safe-r1",
        )
        for field, value in (
            ("plan_sha256", "0" * 64),
            ("prerequisites_sha256", "0" * 64),
            ("source_tree_sha256", "0" * 64),
            ("gpu_indices", [4, 5, 6, 7]),
            ("consumed", True),
        ):
            changed = copy.deepcopy(payload)
            changed[field] = value
            try:
                authorization.validate_authorization(plan, changed)
            except ValueError:
                pass
            else:
                raise AssertionError(f"tampered {field} was accepted")

        changed = copy.deepcopy(payload)
        changed["case_port_pairs"][1]["dist_port"] = (
            changed["case_port_pairs"][0]["master_port"]
        )
        try:
            authorization.validate_authorization(plan, changed)
        except ValueError as error:
            assert "authorization" in str(error), str(error)
        else:
            raise AssertionError("duplicate case port was accepted")

        for nonce in ("", "../escape", "with space", "semi;colon"):
            try:
                authorization.produce_authorization(
                    plan=plan,
                    output_path=(
                        Path(temporary) / f"{len(nonce)}.json"
                    ),
                    nonce=nonce,
                )
            except ValueError as error:
                assert "nonce" in str(error), str(error)
            else:
                raise AssertionError("unsafe nonce was accepted")


def test_authorization_requires_exact_70_case_inventory():
    for changed in (
        _plan()["case_commands"][:-1],
        _plan()["case_commands"] + [{
            "case_id": "extra",
            "dist_port": 30000,
            "master_port": 30001,
        }],
    ):
        plan = _plan()
        plan["case_commands"] = changed
        try:
            authorization.produce_authorization(
                plan=plan,
                output_path="/tmp/never-benchmark-authorization.json",
                nonce="case-count-r1",
            )
        except ValueError as error:
            assert "70" in str(error), str(error)
        else:
            raise AssertionError("non-canonical case count was accepted")


def test_consume_authorization_is_atomic_and_single_use():
    plan = _plan()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        active = root / "authorization.json"
        consumed = root / "authorization.consumed.json"
        authorization.produce_authorization(
            plan=plan,
            output_path=active,
            nonce="consume-benchmark-r1",
        )
        result = authorization.consume_authorization(
            plan=plan,
            authorization_path=active,
            consumed_path=consumed,
        )
        assert result["consumed"] is True
        assert not active.exists()
        assert json.loads(consumed.read_text(encoding="utf-8")) == result
        try:
            authorization.consume_authorization(
                plan=plan,
                authorization_path=active,
                consumed_path=root / "again.json",
            )
        except ValueError as error:
            assert "authorization" in str(error), str(error)
        else:
            raise AssertionError("authorization was reused")


def test_consume_claims_active_path_before_consumed_rewrite():
    source = (
        TOOLS
        / (
            "qwen35_tp4_hybrid_prefix_benchmark_"
            "remote_execution_authorization.py"
        )
    ).read_text(encoding="utf-8")
    body = source[source.index("def consume_authorization("):]
    assert body.index(
        "os.replace(authorization_path, consumed_path)"
    ) < body.index("_rewrite_consumed_authorization(")


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 hybrid-prefix benchmark remote execution "
        f"authorization tests passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
