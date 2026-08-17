from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = (
    ROOT
    / "tools"
    / "blockwise_speculative_verifier_gate.py"
)
WORKER_PATH = (
    ROOT
    / "tools"
    / "blockwise_speculative_verifier_worker.py"
)
VERIFIER_PATH = (
    ROOT
    / "tools"
    / "verify_blockwise_speculative_verifier_gate.py"
)
REMOTE_RUNNER_PATH = (
    ROOT
    / "tools"
    / "run_blockwise_speculative_verifier_gate_remote.sh"
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _gate():
    return _load(
        GATE_PATH,
        "blockwise_speculative_verifier_gate_test_module",
    )


def _worker():
    return _load(
        WORKER_PATH,
        "blockwise_speculative_verifier_worker_test_module",
    )


def _verifier():
    return _load(
        VERIFIER_PATH,
        "verify_blockwise_speculative_verifier_gate_test_module",
    )


class _Tokenizer:
    name_or_path = "fake-tokenizer"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [
            (ord(character) % 97) + 1
            for character in text[:17]
        ]


def _movement(**overrides):
    row = {
        key: 0
        for key in _gate().REAL_MOVEMENT_KEYS
    }
    row.update({
        "h2d_copies": 5,
        "h2d_bytes": 1024,
    })
    row.update(overrides)
    return row


def _worker_fixture(policy, context_tokens, batch_size):
    gate = _gate()
    prompts = gate.build_prompt_token_batches(
        _Tokenizer(),
        batch_size=batch_size,
        prompt_tokens=context_tokens,
    )
    outputs = [
        [1000 + prompt_index * 10 + offset for offset in range(8)]
        for prompt_index in range(batch_size)
    ]
    runtime = {
        "proposal_rows": 0,
        "proposed_tokens": 0,
        "accepted_draft_tokens": 0,
        "first_target_callbacks": 0,
        "tail_callbacks": 0,
    }
    if policy == "ngram":
        runtime.update({
            "proposal_rows": batch_size,
            "proposed_tokens": batch_size * 4,
            "accepted_draft_tokens": batch_size * 2,
            "first_target_callbacks": 1,
            "tail_callbacks": 1,
        })
    return {
        "schema_version": gate.SCHEMA_VERSION,
        "classification": gate.CLASSIFICATION,
        "policy": policy,
        "context_tokens": context_tokens,
        "batch_size": batch_size,
        "prompt_rows": prompts,
        "outputs": outputs,
        "runtime": runtime,
        "movement": _movement(),
        "tokenizer_identifier": "fake-tokenizer",
        "dtype": "torch.bfloat16",
        "visible_logical_blocks": (
            context_tokens // 256
        ) * batch_size,
    }


def _worker_fixtures():
    gate = _gate()
    return [
        _worker_fixture(policy, context_tokens, batch_size)
        for policy in gate.POLICIES
        for context_tokens in gate.CONTEXT_TOKENS
        for batch_size in gate.BATCH_SIZES
    ]


def test_prompt_builder_has_exact_lengths_and_stable_digests():
    gate = _gate()
    first = gate.build_prompt_token_batches(
        _Tokenizer(),
        batch_size=4,
        prompt_tokens=16384,
    )
    second = gate.build_prompt_token_batches(
        _Tokenizer(),
        batch_size=4,
        prompt_tokens=16384,
    )

    assert [row["token_count"] for row in first] == [16384] * 4
    assert [row["sha256"] for row in first] == [
        row["sha256"] for row in second
    ]
    assert len({row["sha256"] for row in first}) == 4


def test_worker_key_is_unique_for_policy_context_and_batch():
    gate = _gate()
    keys = {
        gate.worker_key(policy, context_tokens, batch_size)
        for policy in gate.POLICIES
        for context_tokens in gate.CONTEXT_TOKENS
        for batch_size in gate.BATCH_SIZES
    }

    assert len(keys) == 8
    assert "ngram:32768:b4" in keys


def test_movement_subtraction_rejects_invalid_or_decreasing_counters():
    gate = _gate()
    before = _movement(h2d_copies=2)
    after = _movement(h2d_copies=7)

    delta = gate.subtract_counter_summaries(
        before,
        after,
        keys=gate.REAL_MOVEMENT_KEYS,
    )
    assert delta["h2d_copies"] == 5

    for invalid_after in (
        {**after, "h2d_copies": 1},
        {**after, "h2d_copies": 1.5},
        {**after, "h2d_copies": True},
    ):
        with pytest.raises(ValueError):
            gate.subtract_counter_summaries(
                before,
                invalid_after,
                keys=gate.REAL_MOVEMENT_KEYS,
            )


def test_validate_worker_requires_outputs_candidate_runtime_and_movement():
    gate = _gate()
    candidate = _worker_fixture("ngram", 16384, 4)
    validated = gate.validate_worker_result(candidate)
    assert validated["outputs"] == candidate["outputs"]

    mutations = []
    wrong_outputs = copy.deepcopy(candidate)
    wrong_outputs["outputs"][0].pop()
    mutations.append(wrong_outputs)
    for field in (
        "proposed_tokens",
        "accepted_draft_tokens",
        "first_target_callbacks",
        "tail_callbacks",
    ):
        invalid = copy.deepcopy(candidate)
        invalid["runtime"][field] = 0
        mutations.append(invalid)
    no_h2d = copy.deepcopy(candidate)
    no_h2d["movement"]["h2d_copies"] = 0
    mutations.append(no_h2d)
    rejected_copy = copy.deepcopy(candidate)
    rejected_copy["movement"][
        "speculative_residency_rejected_d2h_copies"
    ] = 1
    mutations.append(rejected_copy)

    for invalid in mutations:
        with pytest.raises(ValueError):
            gate.validate_worker_result(invalid)


def test_validate_worker_allows_zero_h2d_when_visible_history_fits():
    gate = _gate()
    worker = _worker_fixture("baseline", 16384, 1)
    worker["movement"]["h2d_copies"] = 0
    worker["movement"]["h2d_bytes"] = 0

    validated = gate.validate_worker_result(worker)

    assert validated["visible_logical_blocks"] == 64
    assert validated["movement"]["h2d_copies"] == 0


def test_artifact_requires_all_eight_cells_parity_and_fixed_classification():
    gate = _gate()
    workers = _worker_fixtures()
    artifact = gate.build_artifact(
        worker_results=workers,
        environment={"host": "test"},
        source_files={"tinyvllm/layers/attention.py": "a" * 64},
    )
    gate.validate_artifact(artifact)
    assert artifact["classification"] == "NOT_PROMOTABLE"
    assert len(artifact["workers"]) == 8
    assert set(artifact["parity"]) == {
        "16384:b1",
        "16384:b4",
        "32768:b1",
        "32768:b4",
    }

    with pytest.raises(ValueError):
        gate.build_artifact(
            worker_results=workers[:-1],
            environment={"host": "test"},
            source_files={
                "tinyvllm/layers/attention.py": "a" * 64,
            },
        )
    invalid_classification = copy.deepcopy(artifact)
    invalid_classification["classification"] = "PROMOTABLE"
    with pytest.raises(ValueError):
        gate.validate_artifact(invalid_classification)
    parity_mismatch = copy.deepcopy(workers)
    parity_mismatch[-1]["outputs"][0][0] += 1
    with pytest.raises(ValueError):
        gate.build_artifact(
            worker_results=parity_mismatch,
            environment={"host": "test"},
            source_files={
                "tinyvllm/layers/attention.py": "a" * 64,
            },
        )


def test_worker_runs_exact_blockwise_configuration_and_lifecycle():
    gate = _gate()
    worker = _worker()
    engine_calls = []
    generation_calls = []

    class FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeRuntime:
        def __init__(self, adapter):
            self.adapter = adapter

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeEngine:
        def __init__(self):
            self.tokenizer = _Tokenizer()
            self.config = type(
                "Config",
                (),
                {"dtype": "torch.bfloat16"},
            )()
            self.runtime = None
            self.clear_calls = 0
            self.exit_calls = 0
            self.counters = _movement(
                h2d_copies=0,
                h2d_bytes=0,
            )

        def activate_speculative_runtime(self, runtime):
            self.runtime = runtime

        def clear_reusable_prefix_cache(self):
            self.clear_calls += 1

        def kv_offload_summaries(self, timeout_s):
            assert timeout_s == 60.0
            return (dict(self.counters),)

        def exit(self):
            self.exit_calls += 1

    engine = FakeEngine()

    def engine_factory(model_path, **kwargs):
        engine_calls.append((model_path, kwargs))
        return engine

    def run_generation_fn(
        *,
        engine,
        prompt_rows,
        sampling_params,
        expected_output_tokens,
        synchronize,
    ):
        generation_calls.append({
            "prompt_rows": copy.deepcopy(prompt_rows),
            "sampling": dict(sampling_params.kwargs),
            "expected_output_tokens": expected_output_tokens,
        })
        synchronize()
        engine.counters["h2d_copies"] += 3
        engine.counters["h2d_bytes"] += 3072
        runtime = {
            "proposal_rows": 0,
            "proposed_tokens": 0,
            "accepted_draft_tokens": 0,
            "first_target_callbacks": 0,
            "tail_callbacks": 0,
        }
        if engine.runtime is not None:
            runtime.update({
                "proposal_rows": len(prompt_rows),
                "proposed_tokens": len(prompt_rows) * 4,
                "accepted_draft_tokens": len(prompt_rows) * 2,
                "first_target_callbacks": 1,
                "tail_callbacks": 1,
            })
        return {
            "outputs": [
                list(range(expected_output_tokens))
                for _ in prompt_rows
            ],
            "runtime": runtime,
        }

    result = worker.run_policy_cell(
        model_path="/models/Qwen3-0.6B",
        policy="ngram",
        context_tokens=32768,
        batch_size=4,
        engine_factory=engine_factory,
        sampling_params_type=FakeSamplingParams,
        runtime_type=FakeRuntime,
        adapter_type=FakeAdapter,
        synchronize=lambda: None,
        run_generation_fn=run_generation_fn,
    )

    assert engine_calls == [(
        "/models/Qwen3-0.6B",
        {
            "tensor_parallel_size": 1,
            "enforce_eager": True,
            "max_model_len": 33024,
            "max_num_batched_tokens": 132096,
            "max_num_seqs": 4,
            "max_num_prefill_tokens_per_step": 1024,
            "chunked_prefill_decode_first": False,
            "chunked_prefill_mixed_batch": False,
            "kv_offload_mvp0": True,
            "kv_offload_gpu_blocks": 68,
            "kv_offload_logical_blocks": 640,
            "kv_offload_blockwise_decode": True,
            "kv_offload_blockwise_prefill": True,
            "kv_offload_blockwise_blocks": 8,
        },
    )]
    assert len(generation_calls) == 2
    assert generation_calls[0]["sampling"] == {
        "temperature": 0.0,
        "max_tokens": 8,
        "ignore_eos": True,
    }
    assert engine.clear_calls == 1
    assert engine.exit_calls == 1
    assert isinstance(engine.runtime, FakeRuntime)
    assert engine.runtime.adapter.kwargs == {
        "ngram_size": 3,
        "max_proposal_tokens": 4,
    }
    assert result["movement"]["h2d_copies"] == 3
    assert result["movement"]["h2d_bytes"] == 3072
    assert result["runtime"]["proposed_tokens"] == 16
    assert result["visible_logical_blocks"] == 512
    gate.validate_worker_result(result)


def test_orchestrate_launches_eight_cells_and_writes_valid_artifact(
    tmp_path,
):
    gate = _gate()
    commands = []

    def worker_runner(command, *, log_path, cwd):
        commands.append(list(command))
        policy = command[command.index("--policy") + 1]
        context_tokens = int(
            command[command.index("--context-tokens") + 1]
        )
        batch_size = int(
            command[command.index("--batch-size") + 1]
        )
        output = Path(command[command.index("--out") + 1])
        gate.atomic_write_json(
            output,
            _worker_fixture(
                policy,
                context_tokens,
                batch_size,
            ),
        )
        Path(log_path).write_text("ok\n", encoding="utf-8")
        assert Path(cwd) == tmp_path
        return 0

    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    output = tmp_path / "result.json"
    artifact = gate.run_gate(
        model_path="/models/Qwen3-0.6B",
        output_path=output,
        repo_root=tmp_path,
        worker_script=WORKER_PATH,
        worker_runner=worker_runner,
        python_executable="/env/bin/python",
        source_files=("source.py",),
        environment={"host": "test"},
    )

    assert len(commands) == 8
    assert {
        (
            command[command.index("--policy") + 1],
            int(command[
                command.index("--context-tokens") + 1
            ]),
            int(command[
                command.index("--batch-size") + 1
            ]),
        )
        for command in commands
    } == {
        (policy, context_tokens, batch_size)
        for policy in gate.POLICIES
        for context_tokens in gate.CONTEXT_TOKENS
        for batch_size in gate.BATCH_SIZES
    }
    assert output.is_file()
    gate.validate_artifact(artifact)


def test_orchestrate_rejects_worker_failure_or_missing_output(
    tmp_path,
):
    gate = _gate()
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(RuntimeError):
        gate.run_gate(
            model_path="/models/Qwen3-0.6B",
            output_path=tmp_path / "failed.json",
            repo_root=tmp_path,
            worker_script=WORKER_PATH,
            worker_runner=lambda *args, **kwargs: 7,
            source_files=("source.py",),
        )

    with pytest.raises(RuntimeError):
        gate.run_gate(
            model_path="/models/Qwen3-0.6B",
            output_path=tmp_path / "missing.json",
            repo_root=tmp_path,
            worker_script=WORKER_PATH,
            worker_runner=lambda *args, **kwargs: 0,
            source_files=("source.py",),
        )


def test_verifier_passes_then_rejects_source_hash_drift(tmp_path):
    gate = _gate()
    verifier = _verifier()
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    artifact = gate.build_artifact(
        worker_results=_worker_fixtures(),
        environment={"host": "test"},
        source_files={
            "source.py": gate.sha256_file(source),
        },
    )
    artifact_path = tmp_path / "result.json"
    receipt_path = tmp_path / "verify.json"
    gate.atomic_write_json(artifact_path, artifact)

    receipt = verifier.verify_artifact(
        artifact_path,
        tmp_path,
        output_path=receipt_path,
    )
    assert receipt["status"] == "PASS"
    assert set(receipt["cells"]) == {
        "16384:b1",
        "16384:b4",
        "32768:b1",
        "32768:b4",
    }
    assert receipt_path.is_file()

    source.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        verifier.verify_artifact(
            artifact_path,
            tmp_path,
        )


def test_remote_runner_has_fixed_authority_and_always_downloads():
    source = REMOTE_RUNNER_PATH.read_text(
        encoding="utf-8"
    )
    required = (
        "sitian@10.232.195.203",
        "FILE:/Users/bytedance/krb5cc_sitian",
        "ControlMaster=no",
        "ControlPath=none",
        "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python",
        "/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B",
        "/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge",
        "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'",
        "py_compile",
        "verify.remote.json",
        "verify.json",
        "set +e",
        "rsync",
        "nohup",
        "campaign.pid",
        "campaign.status",
        "campaign.exit_code",
        "campaign.sh.tmp",
        "stale RUNNING campaign",
    )
    for value in required:
        assert value in source
    assert source.count("rsync") >= 3
    assert "CUDA_VISIBLE_DEVICES=\"${CUDA_VISIBLE_DEVICES:-7}\"" in source
    assert "ssh -S" not in source
    assert '(cd "${REPO_ROOT}"' in source
    assert "./tools/blockwise_speculative_verifier_gate.py" in source


def test_remote_runner_persists_recovers_and_fails_closed(
    tmp_path,
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    remote_repo = tmp_path / "remote-repo"
    local_out = tmp_path / "local-out"
    campaign_count = tmp_path / "campaign-count"
    fake_python = fake_bin / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-m" && "${2:-}" == "py_compile" ]]; then
  exit 0
fi
output=""
previous=""
for argument in "$@"; do
  if [[ "${previous}" == "--out" || "${previous}" == "--output" ]]; then
    output="${argument}"
  fi
  previous="${argument}"
done
if [[ "$*" == *"blockwise_speculative_verifier_gate.py run"* ]]; then
  count="$(cat "${FAKE_CAMPAIGN_COUNT_FILE}" 2>/dev/null || printf 0)"
  printf '%s\\n' "$((count + 1))" > "${FAKE_CAMPAIGN_COUNT_FILE}"
  if [[ -n "${FAKE_CAMPAIGN_SLEEP_SECONDS:-}" ]]; then
    sleep "${FAKE_CAMPAIGN_SLEEP_SECONDS}"
  fi
  mkdir -p "$(dirname "${output}")"
  printf '%s\\n' '{"status":"FAKE"}' > "${output}"
  exit 0
fi
if [[ "$*" == *"verify_blockwise_speculative_verifier_gate.py"* ]]; then
  mkdir -p "$(dirname "${output}")"
  printf '%s\\n' '{"status":"PASS"}' > "${output}"
  exit 0
fi
exit 2
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    (fake_bin / "python3").symlink_to(fake_python)

    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
remote_command="${!#}"
if [[ -n "${FAKE_FAIL_POLL_ONCE_FILE:-}" ]] &&
   [[ "${remote_command}" == *"printf '%s:%s"* ]] &&
   [[ ! -e "${FAKE_FAIL_POLL_ONCE_FILE}" ]]; then
  touch "${FAKE_FAIL_POLL_ONCE_FILE}"
  exit 255
fi
bash -c "${remote_command}"
""",
        encoding="utf-8",
    )
    fake_ssh.chmod(0o755)

    fake_rsync = fake_bin / "rsync"
    fake_rsync.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
arguments=("$@")
count="${#arguments[@]}"
source_path="${arguments[$((count - 2))]}"
destination_path="${arguments[$((count - 1))]}"
if [[ "${source_path}" == *:* ]]; then
  source_path="${source_path#*:}"
  mkdir -p "${destination_path}"
  cp -R "${source_path}/." "${destination_path}/"
fi
""",
        encoding="utf-8",
    )
    fake_rsync.chmod(0o755)

    environment = os.environ.copy()
    environment.update({
        "PATH": f"{fake_bin}:{environment['PATH']}",
        "REMOTE_HOST": "fake@remote",
        "CONTROL_SOCKET": str(tmp_path / "control.sock"),
        "REMOTE_PYTHON": str(fake_python),
        "MODEL_PATH": str(tmp_path / "fake-model"),
        "REMOTE_REPO": str(remote_repo),
        "RUN_TAG": "persistent-test",
        "LOCAL_OUT": str(local_out),
        "POLL_INTERVAL_SECONDS": "0.01",
        "FAKE_CAMPAIGN_COUNT_FILE": str(campaign_count),
    })

    first = subprocess.run(
        ["bash", str(REMOTE_RUNNER_PATH)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    remote_out = (
        remote_repo
        / "artifacts"
        / "blockwise_speculative_verifier"
        / "persistent-test"
    )
    assert campaign_count.read_text(encoding="utf-8").strip() == "1"
    assert (
        remote_out / "campaign.status"
    ).read_text(encoding="utf-8").strip() == "COMPLETE"
    assert (
        remote_out / "campaign.exit_code"
    ).read_text(encoding="utf-8").strip() == "0"
    assert (local_out / "result.json").is_file()
    assert (local_out / "verify.remote.json").is_file()
    assert (local_out / "verify.json").is_file()

    resumed = subprocess.run(
        ["bash", str(REMOTE_RUNNER_PATH)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert campaign_count.read_text(encoding="utf-8").strip() == "1"
    assert "campaign already terminal" in resumed.stdout

    disconnected_out = tmp_path / "disconnected-out"
    failed_poll = tmp_path / "failed-poll"
    environment.update({
        "RUN_TAG": "disconnect-test",
        "LOCAL_OUT": str(disconnected_out),
        "FAKE_FAIL_POLL_ONCE_FILE": str(failed_poll),
        "FAKE_CAMPAIGN_SLEEP_SECONDS": "0.2",
    })
    disconnected = subprocess.run(
        ["bash", str(REMOTE_RUNNER_PATH)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert disconnected.returncode == 255
    assert "remote poll unavailable" in disconnected.stderr
    disconnected_remote_out = (
        remote_repo
        / "artifacts"
        / "blockwise_speculative_verifier"
        / "disconnect-test"
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status_path = disconnected_remote_out / "campaign.status"
        if (
            status_path.is_file()
            and status_path.read_text(
                encoding="utf-8"
            ).strip() == "COMPLETE"
        ):
            break
        time.sleep(0.01)
    else:
        pytest.fail("detached campaign did not complete")
    assert campaign_count.read_text(encoding="utf-8").strip() == "2"

    recovered = subprocess.run(
        ["bash", str(REMOTE_RUNNER_PATH)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert recovered.returncode == 0, recovered.stderr
    assert campaign_count.read_text(encoding="utf-8").strip() == "2"
    assert "campaign already terminal" in recovered.stdout
    assert (disconnected_out / "result.json").is_file()
    assert (disconnected_out / "verify.remote.json").is_file()
    assert (disconnected_out / "verify.json").is_file()

    environment.pop("FAKE_FAIL_POLL_ONCE_FILE")
    environment.pop("FAKE_CAMPAIGN_SLEEP_SECONDS")
    environment["RUN_TAG"] = "persistent-test"
    environment["LOCAL_OUT"] = str(local_out)
    (remote_out / "campaign.status").write_text(
        "RUNNING\n",
        encoding="utf-8",
    )
    (remote_out / "campaign.pid").write_text(
        "999999999\n",
        encoding="utf-8",
    )
    stale = subprocess.run(
        ["bash", str(REMOTE_RUNNER_PATH)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert stale.returncode == 125, stale.stderr
    assert campaign_count.read_text(encoding="utf-8").strip() == "2"
    assert (
        remote_out / "campaign.status"
    ).read_text(encoding="utf-8").strip() == "FAILED"
    assert (
        remote_out / "campaign.exit_code"
    ).read_text(encoding="utf-8").strip() == "125"
