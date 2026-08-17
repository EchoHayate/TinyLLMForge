from dataclasses import replace

import pytest

import qwen35_tp4_request_e2e64_runner as runner


EXPECTED_CASE_IDS = {
    "w2_long_reuse__warmup__r0__recompute",
    "w2_long_reuse__warmup__r0__exact_restore",
    *{
        f"w2_long_reuse__measured__r{repetition}__{policy}"
        for repetition in range(5)
        for policy in ("recompute", "exact_restore")
    },
}


@pytest.mark.parametrize("source_name", ("r620", "r631"))
def test_build_commands_produces_canonical_64_token_matrix(source_name):
    source = runner.SOURCES[source_name]

    commands = runner.build_commands(source)

    assert len(commands) == 12
    assert {row["case_id"] for row in commands} == EXPECTED_CASE_IDS
    for row in commands:
        argv = row["argv"]
        assert "--profile" in argv
        assert "--generated-tokens-override" not in argv
        assert "--decode-internal-profile" not in argv
        assert source.tag in " ".join(argv)
        assert source.source_tree_sha256 in argv
        assert runner.TEMPLATE_TAG not in " ".join(argv)
        assert runner.TEMPLATE_SOURCE_SHA256 not in argv


def test_validate_commands_rejects_short_output_override():
    source = runner.SOURCES["r631"]
    commands = runner.build_commands(source)
    commands[0]["argv"].extend([
        "--generated-tokens-override",
        "8",
    ])

    with pytest.raises(
        ValueError,
        match="generated-tokens-override",
    ):
        runner.validate_commands(commands, source)


def test_validate_commands_rejects_decode_internal_profile():
    source = runner.SOURCES["r620"]
    commands = runner.build_commands(source)
    commands[-1]["argv"].append("--decode-internal-profile")

    with pytest.raises(
        ValueError,
        match="decode-internal-profile",
    ):
        runner.validate_commands(commands, source)


def test_build_dry_run_payload_freezes_source_and_resource_policy():
    source = runner.SOURCES["r620"]

    payload = runner.build_dry_run_payload(source)

    assert payload["run_tag"] == source.tag
    assert (
        payload["source_tree_sha256"]
        == source.source_tree_sha256
    )
    assert (
        payload["source_tar_sha256"]
        == source.source_tar_sha256
    )
    assert payload["generated_tokens_per_request"] == 64
    assert payload["requests_per_case"] == 4
    assert payload["selected_gpus"] == [2, 4, 5, 6]
    assert payload["minimum_free_bytes"] == 26843545600
    assert payload["maximum_utilization_percent"] == 10
    assert len(payload["commands"]) == 12


def test_validate_source_rejects_unexpected_attempt_tag():
    source = replace(
        runner.SOURCES["r620"],
        tag=runner.SOURCES["r631"].tag,
    )

    with pytest.raises(ValueError, match="source name"):
        runner.validate_source(source)

