import importlib.util
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTER_PATH = REPO_ROOT / "tinyvllm" / "speculative" / "router.py"
SPEC = importlib.util.spec_from_file_location(
    "speculation_router_under_test",
    os.fspath(ROUTER_PATH),
)
router = importlib.util.module_from_spec(SPEC)
sys.modules["speculation_router_under_test"] = router
SPEC.loader.exec_module(router)

choose_speculation_route = router.choose_speculation_route
route_to_dict = router.route_to_dict


def test_finished_precedes_every_other_decision():
    route = choose_speculation_route(
        draft_len=8,
        finished=True,
        remaining_output_budget=16,
        native_compatible=True,
    )
    assert route.name == "baseline_finished"


def test_output_budget_precedes_short_draft():
    route = choose_speculation_route(
        draft_len=1,
        finished=False,
        remaining_output_budget=0,
        native_compatible=True,
    )
    assert route.name == "baseline_output_budget"


def test_zero_and_one_token_drafts_use_baseline():
    for draft_len in (0, 1):
        route = choose_speculation_route(
            draft_len=draft_len,
            finished=False,
            remaining_output_budget=16,
            native_compatible=True,
        )
        assert route.name == "baseline_short_draft"


def test_compatible_multi_token_draft_uses_native():
    route = choose_speculation_route(
        draft_len=4,
        finished=False,
        remaining_output_budget=16,
        native_compatible=True,
    )
    assert route.name == "native_multi_token"


def test_controlled_incompatibility_fails_closed():
    try:
        choose_speculation_route(
            draft_len=4,
            finished=False,
            remaining_output_budget=16,
            native_compatible=False,
            compatibility_reason="kv_offload_mvp0",
            allow_incompatible_fallback=False,
        )
    except ValueError as exc:
        assert "kv_offload_mvp0" in str(exc)
    else:
        raise AssertionError("expected fail-closed incompatibility")


def test_real_source_incompatibility_records_fallback():
    route = choose_speculation_route(
        draft_len=4,
        finished=False,
        remaining_output_budget=16,
        native_compatible=False,
        compatibility_reason="kv_offload_mvp0",
        allow_incompatible_fallback=True,
    )
    assert route_to_dict(route) == {
        "name": "baseline_incompatible",
        "draft_len": 4,
        "native_compatible": False,
        "fallback_reason": "kv_offload_mvp0",
    }


def test_negative_draft_length_is_rejected():
    try:
        choose_speculation_route(
            draft_len=-1,
            finished=False,
            remaining_output_budget=16,
            native_compatible=True,
        )
    except ValueError as exc:
        assert "draft_len" in str(exc)
    else:
        raise AssertionError("expected negative draft length rejection")


def main():
    test_finished_precedes_every_other_decision()
    test_output_budget_precedes_short_draft()
    test_zero_and_one_token_drafts_use_baseline()
    test_compatible_multi_token_draft_uses_native()
    test_controlled_incompatibility_fails_closed()
    test_real_source_incompatibility_records_fallback()
    test_negative_draft_length_is_rejected()
    print("speculation router tests passed")


if __name__ == "__main__":
    main()
