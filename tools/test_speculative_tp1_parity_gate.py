from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "tools" / "speculative_tp1_parity_gate.py"
VERIFY_PATH = (
    ROOT / "tools" / "verify_speculative_tp1_parity_gate.py"
)
REMOTE_SCRIPT_PATH = (
    ROOT
    / "tools"
    / "run_speculative_tp1_parity_gate_remote.sh"
)


def _load_module(name: str, path: Path):
    assert path.exists(), f"missing module: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _gate():
    return _load_module(
        "speculative_tp1_parity_gate_test_module",
        GATE_PATH,
    )


def _verifier():
    return _load_module(
        "verify_speculative_tp1_parity_gate_test_module",
        VERIFY_PATH,
    )


OBSERVATIONS = [
    {
        "speculative_selected_seq_ids": [7],
        "speculative_proposal_token_counts": {7: 3},
        "speculative_proposal_row_count": 1,
        "speculative_accepted_draft_token_counts": {7: 2},
        "speculative_first_target_callback_count": 1,
        "speculative_fixed_q_group_count": 1,
    },
    {
        "speculative_selected_seq_ids": [7],
        "speculative_proposal_token_counts": {7: 2},
        "speculative_proposal_row_count": 1,
        "speculative_accepted_draft_token_counts": {7: 0},
        "speculative_first_target_callback_count": 1,
        "speculative_fixed_q_group_count": 1,
    },
]

MOVEMENT = {
    "h2d_copies": 4,
    "d2h_copies": 2,
    "h2d_bytes": 4096,
    "d2h_bytes": 2048,
    "copy_waits": 3,
    "evictions": 1,
    "evict_clean": 1,
    "evict_dirty": 0,
}

RESIDENCY = {
    "speculative_residency_prepares": 2,
    "speculative_residency_precommits": 2,
    "speculative_residency_seals": 2,
    "speculative_residency_rollbacks": 0,
    "speculative_residency_committed_blocks": 1,
    "speculative_residency_rejected_blocks": 1,
    "speculative_residency_rejected_d2h_copies": 0,
}


def _environment_fixture():
    return {
        "model_path": "/models/Qwen3-0.6B",
        "model_identifier": "Qwen3-0.6B",
        "tokenizer_identifier": "Qwen3-0.6B",
        "tensor_parallel_size": 1,
        "temperature": 0.0,
        "kv_offload_mvp0": True,
        "dtype": "bfloat16",
        "device_name": "A100",
        "python_version": "3.12.1",
        "torch_version": "2.4.1",
        "command": [
            "python",
            "tools/speculative_tp1_parity_gate.py",
            "run",
        ],
    }


def _summary_fixture():
    return {
        "selected_rows": 2,
        "proposal_rows": 2,
        "proposed_tokens": 5,
        "accepted_draft_tokens": 2,
        "first_target_callbacks": 2,
        "tail_callbacks": 2,
        "target_invocations": 4,
        "acceptance_rate": 0.4,
        "accepted_tokens_per_target_invocation": 0.5,
    }


def _artifact():
    gate = _gate()
    return gate.build_parity_artifact(
        baseline={
            "outputs": [[10, 11, 12]],
            "prompt_token_ids": [[1, 2, 3]],
            "elapsed_s": 1.0,
            "movement": MOVEMENT,
        },
        speculative={
            "outputs": [[10, 11, 12]],
            "prompt_token_ids": [[1, 2, 3]],
            "summary": _summary_fixture(),
            "elapsed_s": 0.9,
            "movement": MOVEMENT,
            "residency": RESIDENCY,
        },
        environment=_environment_fixture(),
        source_files={
            "tinyvllm/engine/llm_engine.py": "a" * 64,
        },
    )


def test_aggregate_uses_direct_production_observations():
    summary = _gate().aggregate_speculative_observations(
        OBSERVATIONS
    )

    assert summary == _summary_fixture()


@pytest.mark.parametrize(
    "mutation",
    (
        lambda row: row.pop(
            "speculative_proposal_token_counts"
        ),
        lambda row: row.__setitem__(
            "speculative_proposal_row_count",
            -1,
        ),
        lambda row: row.__setitem__(
            "speculative_first_target_callback_count",
            True,
        ),
        lambda row: row.__setitem__(
            "speculative_accepted_draft_token_counts",
            {7: 4},
        ),
    ),
)
def test_aggregate_rejects_missing_or_invalid_direct_counts(
    mutation,
):
    rows = copy.deepcopy(OBSERVATIONS)
    mutation(rows[0])

    with pytest.raises(ValueError):
        _gate().aggregate_speculative_observations(rows)


def test_artifact_requires_exact_token_parity_and_real_selection():
    artifact = _artifact()

    result = _gate().validate_parity_artifact(artifact)

    assert result["status"] == "PASS"
    assert result["output_sequences"] == 1
    assert result["proposed_tokens"] == 5
    assert result["accepted_draft_tokens"] == 2
    assert result["target_invocations"] == 4
    assert artifact["schema_version"] == 2
    assert artifact["environment"]["kv_offload_mvp0"] is True
    assert artifact["speculative"]["residency"] == RESIDENCY


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    (
        ("baseline", "h2d_copies", None, "h2d_copies"),
        ("baseline", "d2h_bytes", -1, "d2h_bytes"),
        ("speculative", "copy_waits", True, "copy_waits"),
        ("speculative", "evictions", 1.5, "evictions"),
    ),
)
def test_artifact_rejects_invalid_real_movement(
    section,
    key,
    value,
    message,
):
    artifact = _artifact()
    if value is None:
        artifact[section]["movement"].pop(key)
    else:
        artifact[section]["movement"][key] = value

    with pytest.raises(ValueError, match=message):
        _gate().validate_parity_artifact(artifact)


def test_artifact_requires_positive_residency_phases_and_zero_rejected_d2h():
    for key in (
        "speculative_residency_prepares",
        "speculative_residency_precommits",
        "speculative_residency_seals",
    ):
        artifact = _artifact()
        artifact["speculative"]["residency"][key] = 0
        with pytest.raises(ValueError, match=key):
            _gate().validate_parity_artifact(artifact)

    artifact = _artifact()
    artifact["speculative"]["residency"][
        "speculative_residency_rejected_d2h_copies"
    ] = 1
    with pytest.raises(ValueError, match="rejected_d2h"):
        _gate().validate_parity_artifact(artifact)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (
            ("speculative", "outputs"),
            [[10, 99, 12]],
            "token parity",
        ),
        (
            ("speculative", "summary", "selected_rows"),
            0,
            "selected",
        ),
        (
            ("speculative", "summary", "proposal_rows"),
            0,
            "proposal",
        ),
        (
            (
                "speculative",
                "summary",
                "first_target_callbacks",
            ),
            0,
            "first-target",
        ),
        (
            ("speculative", "summary", "tail_callbacks"),
            0,
            "tail",
        ),
        (
            ("environment", "tensor_parallel_size"),
            4,
            "TP1",
        ),
        (
            ("environment", "temperature"),
            0.5,
            "temperature",
        ),
        (
            ("environment", "model_identifier"),
            "",
            "model_identifier",
        ),
    ),
)
def test_artifact_validation_rejects_missing_gate(
    path,
    value,
    message,
):
    artifact = _artifact()
    target = artifact
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        _gate().validate_parity_artifact(artifact)


def test_artifact_rejects_performance_claim_without_controlled_gate():
    artifact = _artifact()
    artifact["performance_improvement_claim"] = True

    with pytest.raises(ValueError, match="performance"):
        _gate().validate_parity_artifact(artifact)


def test_artifact_rejects_malformed_source_hash():
    artifact = _artifact()
    artifact["source_files"][
        "tinyvllm/engine/llm_engine.py"
    ] = "not-a-hash"

    with pytest.raises(ValueError, match="SHA-256"):
        _gate().validate_parity_artifact(artifact)


def test_independent_verifier_accepts_matching_source_hash(
    tmp_path,
):
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    artifact = _artifact()
    artifact["source_files"] = {
        "source.py": hashlib.sha256(
            source.read_bytes()
        ).hexdigest(),
    }
    artifact_path = tmp_path / "result.json"
    artifact_path.write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )

    result = _verifier().verify_artifact(
        artifact_path=artifact_path,
        repo_root=tmp_path,
    )

    assert result["status"] == "PASS"
    assert result["proposed_tokens"] == 5


def test_independent_verifier_rejects_source_tamper(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    artifact = _artifact()
    artifact["source_files"] = {
        "source.py": hashlib.sha256(
            source.read_bytes()
        ).hexdigest(),
    }
    artifact_path = tmp_path / "result.json"
    artifact_path.write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )
    source.write_text("value = 2\n", encoding="utf-8")

    with pytest.raises(
        Exception,
        match="source hash mismatch",
    ):
        _verifier().verify_artifact(
            artifact_path=artifact_path,
            repo_root=tmp_path,
        )


class _FakeSamplingParams:
    def __init__(
        self,
        *,
        temperature,
        max_tokens,
        ignore_eos,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos


class _FakeAdapter:
    def __init__(
        self,
        *,
        ngram_size,
        max_proposal_tokens,
    ):
        self.ngram_size = ngram_size
        self.max_proposal_tokens = max_proposal_tokens


class _FakeRuntime:
    def __init__(self, draft_adapter):
        self.draft_adapter = draft_adapter


class _FakeTokenizer:
    name_or_path = "fake-tokenizer"

    def encode(self, prompt):
        return [ord(character) for character in prompt]


class _FakeEngine:
    def __init__(self, outputs, observations):
        self.tokenizer = _FakeTokenizer()
        self.config = type(
            "Config",
            (),
            {"dtype": "bfloat16"},
        )()
        self._outputs = outputs
        self._observations = list(observations)
        self._step_index = 0
        self.last_step_observation = None
        self.activations = []
        self.requests = []
        self.exit_calls = 0

    def activate_speculative_runtime(self, runtime):
        self.activations.append(runtime)

    def add_request(self, prompt, sampling_params):
        self.requests.append((prompt, sampling_params))

    def is_finished(self):
        return self._step_index >= len(
            self._observations
        )

    def step(self):
        observation = self._observations[
            self._step_index
        ]
        self.last_step_observation = observation
        self._step_index += 1
        if self.is_finished():
            return (
                [
                    (index, list(token_ids))
                    for index, token_ids
                    in enumerate(self._outputs)
                ],
                -1,
            )
        return [], -1

    def exit(self):
        self.exit_calls += 1

    def kv_offload_summaries(self, *, timeout_s):
        assert timeout_s == 60.0
        speculative = any(
            observation[
                "speculative_proposal_row_count"
            ] > 0
            for observation in self._observations
        )
        return ({
            **MOVEMENT,
            **(
                RESIDENCY
                if speculative
                else {
                    key: 0
                    for key in RESIDENCY
                }
            ),
        },)


class _FakeEngineFactory:
    def __init__(self, outputs, observations):
        self.outputs = outputs
        self.observations = observations
        self.instances = []
        self.calls = []

    def __call__(self, model_path, **kwargs):
        self.calls.append((model_path, kwargs))
        engine = _FakeEngine(
            self.outputs,
            self.observations,
        )
        self.instances.append(engine)
        return engine


def _install_fake_runtime_dependencies(gate):
    gate._load_runtime_dependencies = lambda: (
        _FakeSamplingParams,
        _FakeRuntime,
        _FakeAdapter,
    )


def test_run_case_activates_only_speculative_engine():
    gate = _gate()
    _install_fake_runtime_dependencies(gate)
    baseline_factory = _FakeEngineFactory(
        outputs=[[21, 22, 23]],
        observations=[
            {
                "speculative_selected_seq_ids": [],
                "speculative_proposal_token_counts": {},
                "speculative_proposal_row_count": 0,
                "speculative_accepted_draft_token_counts": {},
                "speculative_first_target_callback_count": 0,
                "speculative_fixed_q_group_count": 0,
            },
        ],
    )
    speculative_factory = _FakeEngineFactory(
        outputs=[[21, 22, 23]],
        observations=OBSERVATIONS,
    )

    baseline = gate.run_engine_case(
        engine_factory=baseline_factory,
        model_path="/models/Qwen3-0.6B",
        prompts=("repeat repeat repeat",),
        max_tokens=16,
        activate=False,
        ngram_size=3,
        max_proposal_tokens=4,
    )
    speculative = gate.run_engine_case(
        engine_factory=speculative_factory,
        model_path="/models/Qwen3-0.6B",
        prompts=("repeat repeat repeat",),
        max_tokens=16,
        activate=True,
        ngram_size=3,
        max_proposal_tokens=4,
    )

    baseline_engine = baseline_factory.instances[0]
    speculative_engine = (
        speculative_factory.instances[0]
    )
    assert baseline_engine.activations == []
    assert len(speculative_engine.activations) == 1
    assert baseline["outputs"] == speculative["outputs"]
    assert speculative["summary"]["proposal_rows"] == 2
    assert baseline_engine.exit_calls == 1
    assert speculative_engine.exit_calls == 1


def test_run_case_rejects_speculative_noop_fallback():
    gate = _gate()
    _install_fake_runtime_dependencies(gate)
    factory = _FakeEngineFactory(
        outputs=[[21, 22, 23]],
        observations=[
            {
                "speculative_selected_seq_ids": [],
                "speculative_proposal_token_counts": {},
                "speculative_proposal_row_count": 0,
                "speculative_accepted_draft_token_counts": {},
                "speculative_first_target_callback_count": 0,
                "speculative_fixed_q_group_count": 0,
            },
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="did not execute",
    ):
        gate.run_engine_case(
            engine_factory=factory,
            model_path="/models/Qwen3-0.6B",
            prompts=("repeat repeat repeat",),
            max_tokens=16,
            activate=True,
            ngram_size=3,
            max_proposal_tokens=4,
        )

    assert factory.instances[0].exit_calls == 1


def test_hash_source_files_uses_safe_relative_paths(tmp_path):
    gate = _gate()
    source = tmp_path / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")

    hashes = gate.hash_source_files(
        repo_root=tmp_path,
        source_files=("module.py",),
    )

    assert hashes == {
        "module.py": hashlib.sha256(
            source.read_bytes()
        ).hexdigest(),
    }
    with pytest.raises(ValueError, match="relative"):
        gate.hash_source_files(
            repo_root=tmp_path,
            source_files=("../module.py",),
        )


def test_atomic_artifact_write_rejects_overwrite(tmp_path):
    gate = _gate()
    path = tmp_path / "result.json"
    payload = {"status": "PASS"}

    gate.write_json_atomic(path, payload)

    assert json.loads(path.read_text()) == payload
    assert not path.with_suffix(".json.tmp").exists()
    with pytest.raises(FileExistsError):
        gate.write_json_atomic(path, payload)


def test_run_live_gate_builds_valid_source_bound_artifact(
    tmp_path,
):
    gate = _gate()
    source = tmp_path / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    cases = [
        {
            "outputs": [[10, 11, 12]],
            "prompt_token_ids": [[1, 2, 3]],
            "summary": {
                "selected_rows": 0,
                "proposal_rows": 0,
                "proposed_tokens": 0,
                "accepted_draft_tokens": 0,
                "first_target_callbacks": 0,
                "tail_callbacks": 0,
                "target_invocations": 0,
                "acceptance_rate": 0.0,
                "accepted_tokens_per_target_invocation": 0.0,
            },
            "elapsed_s": 1.0,
            "movement": MOVEMENT,
            "residency": {
                key: 0 for key in RESIDENCY
            },
            "tokenizer_identifier": "tokenizer",
            "dtype": "bfloat16",
        },
        {
            "outputs": [[10, 11, 12]],
            "prompt_token_ids": [[1, 2, 3]],
            "summary": _summary_fixture(),
            "elapsed_s": 0.9,
            "movement": MOVEMENT,
            "residency": RESIDENCY,
            "tokenizer_identifier": "tokenizer",
            "dtype": "bfloat16",
        },
    ]

    def fake_run_engine_case(**kwargs):
        expected_activate = len(cases) == 1
        assert kwargs["activate"] is expected_activate
        return cases.pop(0)

    gate.run_engine_case = fake_run_engine_case
    gate._device_environment = lambda: {
        "device_name": "A100",
        "python_version": "3.12.1",
        "torch_version": "2.4.1",
    }
    output = tmp_path / "result.json"

    artifact = gate.run_live_gate(
        engine_factory=lambda *args, **kwargs: None,
        repo_root=tmp_path,
        model_path="/models/Qwen3-0.6B",
        prompts=("prompt",),
        max_tokens=16,
        ngram_size=3,
        max_proposal_tokens=4,
        output_path=output,
        command=["python", "gate.py", "run"],
        source_files=("module.py",),
    )

    assert artifact["status"] == "PASS"
    assert output.is_file()
    assert gate.validate_parity_artifact(artifact)[
        "status"
    ] == "PASS"


def test_run_live_gate_persists_token_divergence(
    tmp_path,
):
    gate = _gate()
    source = tmp_path / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    cases = [
        {
            "outputs": [[10, 11, 12]],
            "prompt_token_ids": [[1, 2, 3]],
            "summary": {
                "selected_rows": 0,
                "proposal_rows": 0,
                "proposed_tokens": 0,
                "accepted_draft_tokens": 0,
                "first_target_callbacks": 0,
                "tail_callbacks": 0,
                "target_invocations": 0,
                "acceptance_rate": 0.0,
                "accepted_tokens_per_target_invocation": 0.0,
            },
            "elapsed_s": 1.0,
            "movement": MOVEMENT,
            "residency": {
                key: 0 for key in RESIDENCY
            },
            "tokenizer_identifier": "tokenizer",
            "dtype": "bfloat16",
        },
        {
            "outputs": [[10, 99, 12]],
            "prompt_token_ids": [[1, 2, 3]],
            "summary": _summary_fixture(),
            "elapsed_s": 0.9,
            "movement": MOVEMENT,
            "residency": RESIDENCY,
            "tokenizer_identifier": "tokenizer",
            "dtype": "bfloat16",
        },
    ]

    gate.run_engine_case = lambda **kwargs: cases.pop(0)
    gate._device_environment = lambda: {
        "device_name": "A100",
        "python_version": "3.12.1",
        "torch_version": "2.4.1",
    }
    output = tmp_path / "result.json"

    with pytest.raises(
        RuntimeError,
        match="token parity",
    ):
        gate.run_live_gate(
            engine_factory=lambda *args, **kwargs: None,
            repo_root=tmp_path,
            model_path="/models/Qwen3-0.6B",
            prompts=("prompt",),
            max_tokens=16,
            ngram_size=3,
            max_proposal_tokens=4,
            output_path=output,
            command=["python", "gate.py", "run"],
            source_files=("module.py",),
        )

    diagnostic = json.loads(output.read_text())
    assert diagnostic["status"] == "FAIL"
    assert (
        diagnostic["failure_reason"]
        == "exact_token_parity_failed"
    )
    assert diagnostic["baseline"]["outputs"] == [
        [10, 11, 12]
    ]
    assert diagnostic["speculative"]["outputs"] == [
        [10, 99, 12]
    ]


def test_gate_cli_has_main_guard():
    text = GATE_PATH.read_text(encoding="utf-8")

    assert 'if __name__ == "__main__":' in text
    assert "\n    main()\n" in text


def test_remote_script_binds_transport_model_and_gate():
    assert REMOTE_SCRIPT_PATH.is_file()
    text = REMOTE_SCRIPT_PATH.read_text(
        encoding="utf-8"
    )
    required = (
        "sitian@10.232.195.203",
        "/tmp/ssh-sitian-10.232.195.203",
        (
            "/data00/home/sitian/sitian-workspace01/"
            "tllm/env/bin/python"
        ),
        (
            "/data00/home/sitian/sitian-workspace01/"
            ".ms_cache/Qwen/Qwen3-0.6B"
        ),
        "CUDA_VISIBLE_DEVICES=",
        "speculative_tp1_parity_gate.py",
        "verify_speculative_tp1_parity_gate.py",
        "rsync",
        "remote.log",
        "verify.json",
    )
    for value in required:
        assert value in text
    forbidden = (
        "git checkout",
        "git reset",
        "git clean",
        "git stash",
        "git add",
        "git commit",
        "git push",
    )
    for value in forbidden:
        assert value not in text


def test_remote_script_syncs_package_and_downloads_failure_log():
    text = REMOTE_SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert re.search(r"(?m)^  tinyvllm/$", text)
    assert "remote_status=" in text
    assert 'exit "${remote_status}"' in text
    remote_run = text.index(
        'ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" bash -s --'
    )
    download = text.index(
        '"${REMOTE_HOST}:${REMOTE_OUT}/"'
    )
    status_exit = text.index('exit "${remote_status}"')
    assert remote_run < download < status_exit
