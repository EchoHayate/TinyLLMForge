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


contract = _load(
    "qwen35_tp4_hybrid_prefix_contract_for_prerequisite_builder_test",
    "qwen35_tp4_hybrid_prefix_benchmark_contract.py",
)
builder = _load(
    "build_qwen35_tp4_performance_prerequisites",
    "build_qwen35_tp4_performance_prerequisites.py",
)
cached_contract = _load(
    "qwen35_tp4_cached_continuation_contract_for_prerequisite_builder_test",
    "qwen35_tp4_cached_continuation_correctness_contract.py",
)
engine_contract = _load(
    "qwen35_tp4_engine_contract_for_prerequisite_builder_test",
    "qwen35_tp4_engine_correctness_contract.py",
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _root_payloads():
    comparisons = []
    for index, case_id in enumerate(contract.TP4_ROOT_CASE_IDS):
        winner = 100 + index
        runner_up = 200 + index
        comparisons.append({
            "case_id": case_id,
            "native_winner_token_id": winner,
            "native_runner_up_token_id": runner_up,
            "native_winner_margin": 1.0,
            "official_winner_token_id": winner,
            "official_runner_up_token_id": runner_up,
            "official_winner_margin": 1.0,
            "native_topk_token_ids": [winner, runner_up],
            "official_topk_token_ids": [winner, runner_up],
        })
    return (
        {
            "schema_version": (
                contract.TP4_ROOT_CORRECTNESS_SCHEMA_VERSION
            ),
            "run_tag": "tp4-root-logit-run",
            "classification": "PASS",
            "comparison_policy": "registered_logits_strict_allclose",
            "tolerance": {"atol": 2e-5, "rtol": 0.0},
            "prompts": [
                {"case_id": case_id}
                for case_id in contract.TP4_ROOT_CASE_IDS
            ],
            "reference_process": {
                "model_manifest_sha256": (
                    contract.MODEL_MANIFEST_SHA256
                ),
            },
            "comparisons": comparisons,
            "forbidden_counters": {
                "engine": 0,
                "generation": 0,
                "model_runner": 0,
                "sampler": 0,
                "scheduler": 0,
            },
            "claim_boundary": (
                "TP4 root-logit correctness only; no cached decode"
            ),
        },
        {
            "classification": "PASS",
            "case_ids": list(contract.TP4_ROOT_CASE_IDS),
            "ranks": [0, 1, 2, 3],
            "checks": 100,
        },
    )


def test_root_authority_accepts_plan_bound_source_tree():
    artifact, verification = _root_payloads()

    contract.validate_authority_documents(
        "tp4_root_logit",
        artifact,
        verification,
        "a" * 64,
    )


def _cached_rows():
    rows = []
    for workload in cached_contract.WORKLOADS:
        spec = cached_contract.workload_payload(workload)["spec"]
        for request_index in range(spec["continuations"]):
            hit = workload in cached_contract.HIT_WORKLOADS
            rows.append({
                "workload": workload,
                "request_index": request_index,
                "outcome": "continuation",
                "restore_hit": hit,
                "restore_reason": (
                    "exact_hit"
                    if hit
                    else cached_contract.W4_EXPECTED_REASONS[
                        request_index
                    ]
                ),
                "prompt_tokens": (
                    spec["shared_prefix_tokens"]
                    + spec["suffix_tokens"]
                ),
                "reused_tokens": (
                    spec["shared_prefix_tokens"] if hit else 0
                ),
                "executed_prefill_tokens": (
                    spec["suffix_tokens"]
                    if hit
                    else (
                        spec["shared_prefix_tokens"]
                        + spec["suffix_tokens"]
                    )
                ),
                "output_token_ids": [7] * spec["generated_tokens"],
                "reference_output_token_ids": (
                    [7] * spec["generated_tokens"]
                ),
                "logits_max_abs_diff": 0.0,
                "logits_allclose": True,
                "cache_identity_match": True,
                "rank_inventory": [0, 1, 2, 3],
                "process_group_destroyed": True,
                "owned_children_remaining": [],
            })
    return rows


def _cached_payloads(source_tree_sha256):
    rows = _cached_rows()
    classification = cached_contract.classify_rows(rows)
    return (
        {
            "schema_version": cached_contract.SCHEMA_VERSION,
            "classification": "PASS",
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
            "workload_manifest_sha256": (
                cached_contract.WORKLOAD_MANIFEST_SHA256
            ),
            "rows": rows,
        },
        {
            "schema_version": cached_contract.SCHEMA_VERSION,
            "classification": "PASS",
            "source_tree_sha256": source_tree_sha256,
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
            "workload_manifest_sha256": (
                cached_contract.WORKLOAD_MANIFEST_SHA256
            ),
            "checks": classification["checks"],
        },
    )


def _engine_rows():
    rows = []
    for scenario, expected in (
        contract.ENGINE_CORRECTNESS_SCENARIOS.items()
    ):
        (
            scheduler_steps,
            model_runner_calls,
            generated_tokens,
            publication_commits,
            restore_hits,
            restore_misses,
            release_events,
            cache_entries_after,
        ) = expected
        rows.append({
            "scenario": scenario,
            "engine_class": engine_contract.ENGINE_CLASS,
            "model_runner_class": engine_contract.MODEL_RUNNER_CLASS,
            "rank_inventory": [0, 1, 2, 3],
            "ack_ranks": [1, 2, 3],
            "scheduler_steps": scheduler_steps,
            "model_runner_calls": model_runner_calls,
            "output_token_ids": [7] * generated_tokens,
            "reference_output_token_ids": [7] * generated_tokens,
            "publication_commits": publication_commits,
            "restore_hits": restore_hits,
            "restore_misses": restore_misses,
            "release_events": release_events,
            "cache_entries_after": cache_entries_after,
            "cache_identity_match": True,
            "process_group_destroyed": True,
            "rank_exit_codes": [0, 0, 0, 0],
            "owned_children_remaining": [],
        })
    return rows


def _engine_payloads(source_tree_sha256):
    rows = _engine_rows()
    classification = engine_contract.classify_rows(rows)
    assert classification["classification"] == "PASS"
    return (
        {
            "schema_version": engine_contract.SCHEMA_VERSION,
            "classification": "PASS",
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
            "rows": rows,
        },
        {
            "schema_version": engine_contract.SCHEMA_VERSION,
            "classification": "PASS",
            "source_tree_sha256": source_tree_sha256,
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
            "checks": classification["checks"],
        },
    )


def _authority(
    root,
    name,
    source_tree_sha256,
    *,
    forged=False,
):
    source = root / "source" / name
    artifact = source / "artifact.json"
    verification = source / "independent_verification.json"
    provenance = source / "provenance.json"
    plan = source / "execution_plan.json"
    authorization = source / "consumed_authorization.json"
    receipt = source / "execution_receipt.json"
    if forged:
        artifact_payload = verification_payload = {
            "classification": "PASS",
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        }
    elif name == "tp4_root_logit":
        artifact_payload, verification_payload = _root_payloads()
    elif name == "cached_continuation":
        artifact_payload, verification_payload = _cached_payloads(
            source_tree_sha256
        )
    else:
        artifact_payload, verification_payload = _engine_payloads(
            source_tree_sha256
        )
    _write_json(artifact, artifact_payload)
    _write_json(verification, verification_payload)
    _write_json(plan, {"name": name, "kind": "plan"})
    _write_json(
        authorization,
        {"name": name, "kind": "authorization", "consumed": True},
    )
    _write_json(receipt, {"name": name, "kind": "receipt"})
    _write_json(provenance, {
        "schema_version": (
            contract.PREREQUISITE_PROVENANCE_SCHEMA_VERSION
        ),
        "authority_name": name,
        "run_tag": f"{name}-run",
        "binding_kind": "remote_execution_receipt",
        "source_tree_sha256": source_tree_sha256,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "root_logit_receipt_gap": False,
        "plan_path": plan.name,
        "plan_sha256": contract.sha256_file(plan),
        "authorization_path": authorization.name,
        "authorization_sha256": contract.sha256_file(authorization),
        "receipt_path": receipt.name,
        "receipt_sha256": contract.sha256_file(receipt),
    })
    return builder.AuthorityInput(
        name=name,
        run_tag=f"{name}-run",
        source_tree_sha256=source_tree_sha256,
        artifact_path=artifact,
        artifact_sha256=contract.sha256_file(artifact),
        independent_verification_path=verification,
        independent_verification_sha256=(
            contract.sha256_file(verification)
        ),
        provenance_path=provenance,
        provenance_sha256=contract.sha256_file(provenance),
    )


def _inputs(root, *, forged=False):
    return (
        _authority(
            root,
            "tp4_root_logit",
            contract.TP4_ROOT_SOURCE_TREE_SHA256,
            forged=forged,
        ),
        _authority(
            root,
            "cached_continuation",
            "c" * 64,
            forged=forged,
        ),
        _authority(
            root,
            "engine_correctness",
            "d" * 64,
            forged=forged,
        ),
    )


def _expect_value_error(function, fragment):
    try:
        function()
    except ValueError as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(
            f"expected ValueError containing {fragment!r}"
        )


def test_builder_copies_three_authorities_and_emits_valid_bundle():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "bundle"
        result = builder.build_prerequisite_bundle(
            output_dir=output,
            authorities=_inputs(root),
        )

        prerequisite_path = output / "correctness_prerequisites.json"
        status = contract.validate_prerequisites(prerequisite_path)
        assert status.authorized is True
        assert result["classification"] == "PASS"
        assert result["correctness_prerequisites_sha256"] == (
            contract.sha256_file(prerequisite_path)
        )
        assert sorted(
            path.relative_to(output).as_posix()
            for path in output.rglob("*")
            if path.is_file()
        ) == [
            "correctness_prerequisites.json",
            (
                "prerequisites/cached_continuation/"
                "artifact.json"
            ),
            (
                "prerequisites/cached_continuation/"
                "consumed_authorization.json"
            ),
            (
                "prerequisites/cached_continuation/"
                "execution_plan.json"
            ),
            (
                "prerequisites/cached_continuation/"
                "execution_receipt.json"
            ),
            (
                "prerequisites/cached_continuation/"
                "independent_verification.json"
            ),
            "prerequisites/cached_continuation/provenance.json",
            "prerequisites/engine_correctness/artifact.json",
            (
                "prerequisites/engine_correctness/"
                "consumed_authorization.json"
            ),
            "prerequisites/engine_correctness/execution_plan.json",
            (
                "prerequisites/engine_correctness/"
                "execution_receipt.json"
            ),
            (
                "prerequisites/engine_correctness/"
                "independent_verification.json"
            ),
            "prerequisites/engine_correctness/provenance.json",
            "prerequisites/tp4_root_logit/artifact.json",
            (
                "prerequisites/tp4_root_logit/"
                "consumed_authorization.json"
            ),
            "prerequisites/tp4_root_logit/execution_plan.json",
            "prerequisites/tp4_root_logit/execution_receipt.json",
            (
                "prerequisites/tp4_root_logit/"
                "independent_verification.json"
            ),
            "prerequisites/tp4_root_logit/provenance.json",
        ]
        payload = json.loads(prerequisite_path.read_text())
        assert payload["schema_version"] == (
            contract.PREREQUISITE_SCHEMA_VERSION
        )
        for name in builder.AUTHORITY_NAMES:
            assert payload[name]["provenance_path"].endswith(
                f"{name}/provenance.json"
            )
            assert len(payload[name]["provenance_sha256"]) == 64


def test_builder_rejects_legacy_root_directory_only_provenance():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        authorities = list(_inputs(root))
        provenance = authorities[0].provenance_path
        payload = json.loads(provenance.read_text())
        payload.update({
            "binding_kind": "complete_directory_only",
            "root_logit_receipt_gap": True,
            "plan_path": None,
            "plan_sha256": None,
            "authorization_path": None,
            "authorization_sha256": None,
            "receipt_path": None,
            "receipt_sha256": None,
        })
        _write_json(provenance, payload)
        root_authority = authorities[0]
        authorities[0] = builder.AuthorityInput(
            name=root_authority.name,
            run_tag=root_authority.run_tag,
            source_tree_sha256=root_authority.source_tree_sha256,
            artifact_path=root_authority.artifact_path,
            artifact_sha256=root_authority.artifact_sha256,
            independent_verification_path=(
                root_authority.independent_verification_path
            ),
            independent_verification_sha256=(
                root_authority.independent_verification_sha256
            ),
            provenance_path=provenance,
            provenance_sha256=contract.sha256_file(provenance),
        )

        _expect_value_error(
            lambda: builder.build_prerequisite_bundle(
                output_dir=root / "bundle",
                authorities=tuple(authorities),
            ),
            "receipt provenance",
        )


def test_builder_is_deterministic_for_identical_inputs():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        authorities = _inputs(root)
        first = root / "first"
        second = root / "second"
        first_result = builder.build_prerequisite_bundle(
            output_dir=first,
            authorities=authorities,
        )
        second_result = builder.build_prerequisite_bundle(
            output_dir=second,
            authorities=authorities,
        )

        assert first_result == second_result
        assert (
            first / "correctness_prerequisites.json"
        ).read_bytes() == (
            second / "correctness_prerequisites.json"
        ).read_bytes()


def test_builder_rejects_incomplete_duplicate_or_wrong_root_authority():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        authorities = _inputs(root)
        _expect_value_error(
            lambda: builder.build_prerequisite_bundle(
                output_dir=root / "missing",
                authorities=authorities[:2],
            ),
            "authority inventory",
        )
        _expect_value_error(
            lambda: builder.build_prerequisite_bundle(
                output_dir=root / "duplicate",
                authorities=(
                    authorities[0],
                    authorities[0],
                    authorities[2],
                ),
            ),
            "authority inventory",
        )
        wrong_root = builder.AuthorityInput(
            name="tp4_root_logit",
            run_tag="wrong-root",
            source_tree_sha256="e" * 64,
            artifact_path=authorities[0].artifact_path,
            artifact_sha256=authorities[0].artifact_sha256,
            independent_verification_path=(
                authorities[0].independent_verification_path
            ),
            independent_verification_sha256=(
                authorities[0].independent_verification_sha256
            ),
            provenance_path=authorities[0].provenance_path,
            provenance_sha256=authorities[0].provenance_sha256,
        )
        _expect_value_error(
            lambda: builder.build_prerequisite_bundle(
                output_dir=root / "wrong-root",
                authorities=(
                    wrong_root,
                    authorities[1],
                    authorities[2],
                ),
            ),
            "root-logit source tree",
        )


def test_builder_rejects_nonpass_symlink_and_existing_output():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        authorities = list(_inputs(root))
        _write_json(
            authorities[1].artifact_path,
            {
                "classification": "NO_GO",
                "model_manifest_sha256": (
                    contract.MODEL_MANIFEST_SHA256
                ),
            },
        )
        changed = authorities[1]
        authorities[1] = builder.AuthorityInput(
            name=changed.name,
            run_tag=changed.run_tag,
            source_tree_sha256=changed.source_tree_sha256,
            artifact_path=changed.artifact_path,
            artifact_sha256=contract.sha256_file(
                changed.artifact_path
            ),
            independent_verification_path=(
                changed.independent_verification_path
            ),
            independent_verification_sha256=(
                changed.independent_verification_sha256
            ),
            provenance_path=changed.provenance_path,
            provenance_sha256=changed.provenance_sha256,
        )
        _expect_value_error(
            lambda: builder.build_prerequisite_bundle(
                output_dir=root / "nonpass",
                authorities=tuple(authorities),
            ),
            "schema",
        )

        authorities = list(_inputs(root / "fresh"))
        target = authorities[2].artifact_path
        link = target.parent / "artifact-link.json"
        link.symlink_to(target)
        authorities[2] = builder.AuthorityInput(
            name=authorities[2].name,
            run_tag=authorities[2].run_tag,
            source_tree_sha256=authorities[2].source_tree_sha256,
            artifact_path=link,
            artifact_sha256=authorities[2].artifact_sha256,
            independent_verification_path=(
                authorities[2].independent_verification_path
            ),
            independent_verification_sha256=(
                authorities[2].independent_verification_sha256
            ),
            provenance_path=authorities[2].provenance_path,
            provenance_sha256=authorities[2].provenance_sha256,
        )
        _expect_value_error(
            lambda: builder.build_prerequisite_bundle(
                output_dir=root / "symlink",
                authorities=tuple(authorities),
            ),
            "regular file",
        )

        existing = root / "existing"
        existing.mkdir()
        _expect_value_error(
            lambda: builder.build_prerequisite_bundle(
                output_dir=existing,
                authorities=_inputs(root / "third"),
            ),
            "already exists",
        )


def test_builder_rejects_authority_hash_tamper():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        authorities = list(_inputs(root))
        original = authorities[1]
        authorities[1] = builder.AuthorityInput(
            name=original.name,
            run_tag=original.run_tag,
            source_tree_sha256=original.source_tree_sha256,
            artifact_path=original.artifact_path,
            artifact_sha256="0" * 64,
            independent_verification_path=(
                original.independent_verification_path
            ),
            independent_verification_sha256=(
                original.independent_verification_sha256
            ),
            provenance_path=original.provenance_path,
            provenance_sha256=original.provenance_sha256,
        )
        _expect_value_error(
            lambda: builder.build_prerequisite_bundle(
                output_dir=root / "tampered",
                authorities=tuple(authorities),
            ),
            "artifact SHA mismatch",
        )


def test_builder_rejects_forged_two_field_pass_documents():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _expect_value_error(
            lambda: builder.build_prerequisite_bundle(
                output_dir=root / "forged",
                authorities=_inputs(root, forged=True),
            ),
            "schema",
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
        "qwen35 TP4 performance prerequisite builder tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
