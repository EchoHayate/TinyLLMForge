"""Dependency-light tests for the SAM drafter canonical gate."""

from __future__ import annotations

import importlib.util
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_GATE_PATH = os.path.join(_THIS_DIR, "sam_drafter_gate.py")
_SPEC = importlib.util.spec_from_file_location(
    "sam_drafter_gate_under_test",
    _GATE_PATH,
)
gate = importlib.util.module_from_spec(_SPEC)
sys.modules["sam_drafter_gate_under_test"] = gate
_SPEC.loader.exec_module(gate)


def test_prompt_bank_has_five_stable_classes():
    assert [item["name"] for item in gate.PROMPT_BANK] == [
        "natural_prose",
        "structured_code_like",
        "repeated_long_context",
        "transition_heavy",
        "prompt_copy_retrieval",
    ]
    assert {item["workload_class"] for item in gate.PROMPT_BANK} == {
        "natural",
        "structured",
        "high_repeat",
        "transition_heavy",
        "prompt_copy",
    }
    for prompt in gate.PROMPT_BANK:
        assert prompt["prompt_sha256"] == gate.sha256_text(prompt["prompt"])


def test_run_specs_are_175_unique_rows_for_canonical():
    specs = gate.build_run_specs(repetitions=7, base_seed=20260715)
    assert len(specs) == 175
    assert len({item["run_key"] for item in specs}) == 175
    assert {item["policy"] for item in specs} == {
        "baseline",
        "ngram_fixed_k4",
        "ngram_adaptive",
        "sam_fixed_k16",
        "sam_match_aware",
    }
    assert all(item["max_num_seqs"] == 1 for item in specs)


def test_required_upload_paths_cover_all_runtime_imports():
    assert gate.REQUIRED_UPLOAD_PATHS == (
        "tinyvllm",
        "tools/draft_model_schema.py",
        "tools/profile_ngram_commit.py",
        "tools/sam_drafter_gate.py",
    )


def test_profiler_commands_are_policy_specific():
    specs = gate.build_run_specs(repetitions=1, base_seed=7)
    prompt = gate.PROMPT_BANK[0]
    commands = {
        spec["policy"]: gate._profiler_command(
            spec,
            prompt,
            "python3",
            "/model",
            gate.Path("/tmp/process.json"),
        )
        for spec in specs
        if spec["prompt_name"] == prompt["name"]
    }
    assert "--draft-source" not in commands["baseline"]
    for policy in (
        "ngram_fixed_k4",
        "ngram_adaptive",
        "sam_fixed_k16",
        "sam_match_aware",
    ):
        assert "--allow-zero-accept" in commands[policy]
    assert commands["sam_match_aware"][
        commands["sam_match_aware"].index("--draft-source") + 1
    ] == "sam"


def main():
    test_prompt_bank_has_five_stable_classes()
    test_run_specs_are_175_unique_rows_for_canonical()
    test_required_upload_paths_cover_all_runtime_imports()
    test_profiler_commands_are_policy_specific()
    print("sam drafter gate tests passed")


if __name__ == "__main__":
    main()
