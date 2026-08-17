# Atomic Speculative Runtime Activation and TP1 Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one atomic source-agnostic speculative activation boundary and produce an independently verifiable real loaded-model TP1 exact-greedy parity artifact.

**Architecture:** Separate runtime-only preparation from Scheduler compatibility validation, then let `LLMEngine.activate_speculative_runtime()` publish the exact Scheduler selection config and runtime as one rollback-safe operation. Extend existing step observations with direct proposal and target-callback counts, and use those production observations in a dedicated baseline/speculative TP1 runner plus an independent artifact verifier.

**Tech Stack:** Python 3.9/3.12, dataclasses, pytest, TinyLLMForge `LLMEngine`/`Scheduler`/native speculative runtime, JSON and SHA-256 artifacts, Bash, rsync/SSH ControlMaster, Qwen3-0.6B BF16 TP1 on an A100.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, or run `git clean`.
- Generic runtime, Scheduler, verifier, and gate code must not branch on model name or proposal-source name.
- Accepted speculative KV must commit directly; rejected suffixes must roll back without per-token KV replay, rematerialization, or copy.
- Variable proposal lengths remain grouped by distinct fixed Q; do not pad verifier batches.
- Non-greedy and stateful non-KV speculative rows remain fail closed.
- Do not claim TPOT, throughput, memory, TP4, long-context, or offload gains from this gate.
- Do not count simulated KV movement as real offload evidence.
- Dependency-light tests use system `python3`; tests importing local Torch use `/opt/homebrew/bin/python3.12`.
- Remote execution uses `sitian@10.232.195.203`, Python `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`, model `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B`, and SSH ControlMaster `/tmp/ssh-sitian-10.232.195.203`.
- Every task ends with a review checkpoint only; no `git add` or `git commit`.

## File Map

- Modify `tinyvllm/engine/speculative_runtime.py`: runtime-only preparation and exact selection-config derivation.
- Modify `tinyvllm/engine/llm_engine.py`: atomic activation and direct speculative proposal/callback observations.
- Modify `tools/test_engine_speculative_runtime.py`: dependency-light activation transaction tests.
- Modify `tools/test_chunked_prefill.py`: ordinary-path defaults for new observation fields.
- Create `tools/speculative_tp1_parity_gate.py`: live TP1 baseline/speculative runner and artifact builder.
- Create `tools/verify_speculative_tp1_parity_gate.py`: independent artifact and source-integrity verifier.
- Create `tools/test_speculative_tp1_parity_gate.py`: dependency-light artifact, counter, and tamper tests.
- Create `tools/run_speculative_tp1_parity_gate_remote.sh`: reproducible remote sync, execution, download, and verification.
- Modify `docs/superpowers/audits/2026-08-12-generic-inference-optimization-goal-audit.md`: record actual gate evidence and remaining promotion gaps.
- Modify `AGENT_HANDOFF_STATE.md`: preserve commands, artifacts, result, and strict claim boundaries.

---

### Task 1: Separate Runtime Preparation from Published Scheduler Validation

**Files:**
- Modify: `tinyvllm/engine/speculative_runtime.py`
- Test: `tools/test_engine_speculative_runtime.py`

**Interfaces:**
- Consumes: `EngineSpeculativeRuntime`, `DraftCapabilities`, `SpeculativeSelectionConfig`, `model_runner.call`.
- Produces:
  - `build_engine_speculative_selection_config(runtime: EngineSpeculativeRuntime, *, model_runner) -> SpeculativeSelectionConfig`
  - existing `validate_engine_speculative_runtime(runtime, *, scheduler, model_runner) -> EngineSpeculativeRuntime`

- [ ] **Step 1: Add failing tests for runtime-only preparation**

Add the selection module to the dependency-light loader and expose
`SpeculativeSelectionConfig` plus the new builder:

```python
selection_module = _load_module(
    "tinyvllm.engine.speculative_selection",
    ROOT / "tinyvllm" / "engine" / "speculative_selection.py",
)
SpeculativeSelectionConfig = (
    selection_module.SpeculativeSelectionConfig
)
build_engine_speculative_selection_config = (
    runtime_module.build_engine_speculative_selection_config
)
```

Add focused tests:

```python
def test_runtime_preparation_derives_exact_selection_config():
    runtime = EngineSpeculativeRuntime(
        draft_adapter=_Adapter(max_proposal_tokens=4),
        lifecycle=_Lifecycle(),
    )

    config = build_engine_speculative_selection_config(
        runtime,
        model_runner=_model_runner(),
    )

    assert config == SpeculativeSelectionConfig(
        enabled=True,
        max_proposal_tokens=4,
    )


def test_runtime_preparation_rejects_limit_one():
    runtime = EngineSpeculativeRuntime(
        draft_adapter=_Adapter(max_proposal_tokens=1),
    )

    with pytest.raises(
        ValueError,
        match="max_proposal_tokens >= 2",
    ):
        build_engine_speculative_selection_config(
            runtime,
            model_runner=_model_runner(),
        )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python3 -m pytest \
  tools/test_engine_speculative_runtime.py::test_runtime_preparation_derives_exact_selection_config \
  tools/test_engine_speculative_runtime.py::test_runtime_preparation_rejects_limit_one \
  -q
```

Expected: collection or execution fails because
`build_engine_speculative_selection_config` does not exist.

- [ ] **Step 3: Implement runtime-only preparation**

Import the immutable selection config and move adapter, ModelRunner, and
lifecycle validation into the builder:

```python
from tinyvllm.engine.speculative_selection import (
    SpeculativeSelectionConfig,
)


def build_engine_speculative_selection_config(
    runtime: EngineSpeculativeRuntime,
    *,
    model_runner,
) -> SpeculativeSelectionConfig:
    if not isinstance(runtime, EngineSpeculativeRuntime):
        raise ValueError(
            "runtime must be EngineSpeculativeRuntime"
        )
    adapter = runtime.draft_adapter
    capabilities = getattr(adapter, "capabilities", None)
    if not isinstance(capabilities, DraftCapabilities):
        raise ValueError(
            "draft adapter capabilities must be DraftCapabilities"
        )
    if capabilities.supports_batch is not True:
        raise ValueError(
            "draft adapter must support batch proposals"
        )
    if (
        isinstance(capabilities.max_proposal_tokens, bool)
        or not isinstance(
            capabilities.max_proposal_tokens,
            int,
        )
        or capabilities.max_proposal_tokens <= 0
    ):
        raise ValueError(
            "draft adapter proposal limit must be positive"
        )
    if not callable(getattr(adapter, "propose_batch", None)):
        raise ValueError(
            "draft adapter propose_batch must be callable"
        )
    if not callable(getattr(model_runner, "call", None)):
        raise ValueError(
            "model runner callback bridge is unavailable"
        )
    lifecycle = runtime.lifecycle
    if lifecycle is not None:
        for name in (
            "register_sequence",
            "synchronize_verified_history",
            "release_sequence",
        ):
            if not callable(getattr(lifecycle, name, None)):
                raise ValueError(
                    "draft lifecycle must expose callable "
                    f"{name}"
                )
    return SpeculativeSelectionConfig(
        enabled=True,
        max_proposal_tokens=(
            capabilities.max_proposal_tokens
        ),
    )
```

Refactor `validate_engine_speculative_runtime()` to call the builder and only
compare the returned candidate with the Scheduler's published config:

```python
candidate = build_engine_speculative_selection_config(
    runtime,
    model_runner=model_runner,
)
selection = getattr(
    scheduler,
    "speculative_selection_config",
    None,
)
if selection != candidate:
    raise ValueError(
        "Scheduler and draft adapter speculative "
        "selection configs must match"
    )
return runtime
```

- [ ] **Step 4: Run runtime contract tests and verify GREEN**

Run:

```bash
python3 -m pytest tools/test_engine_speculative_runtime.py -q
```

Expected: all tests pass. If old mismatch tests assert a narrower message,
update them to match `"selection configs must match"` without weakening the
state assertion.

- [ ] **Step 5: Review checkpoint**

Run:

```bash
python3 -m py_compile tinyvllm/engine/speculative_runtime.py
git diff --check
```

Expected: both commands succeed. Do not stage or commit.

---

### Task 2: Add Rollback-Safe Atomic Engine Activation

**Files:**
- Modify: `tinyvllm/engine/llm_engine.py`
- Test: `tools/test_engine_speculative_runtime.py`

**Interfaces:**
- Consumes:
  - `build_engine_speculative_selection_config(runtime, *, model_runner)`
  - `Scheduler.install_speculative_selection(config)`
  - `LLMEngine.install_speculative_runtime(runtime)`
- Produces:
  - `LLMEngine.activate_speculative_runtime(runtime: EngineSpeculativeRuntime) -> None`

- [ ] **Step 1: Extend the AST method loader namespace**

Import the builder and selection config into the test namespace:

```python
build_engine_speculative_selection_config = (
    runtime_module.build_engine_speculative_selection_config
)
```

Update `_load_engine_method()`:

```python
namespace = {
    "EngineSpeculativeRuntime": EngineSpeculativeRuntime,
    "validate_engine_speculative_runtime": (
        validate_engine_speculative_runtime
    ),
    "build_engine_speculative_selection_config": (
        build_engine_speculative_selection_config
    ),
}
```

- [ ] **Step 2: Add failing success, idempotence, conflict, and rollback tests**

Use a Scheduler fixture with the real state shape:

```python
class _ActivationScheduler:
    def __init__(self):
        self.speculative_selection_config = (
            SpeculativeSelectionConfig(
                enabled=False,
                max_proposal_tokens=0,
            )
        )
        self._speculative_selection_installed = False
        self.fail_install = False

    def install_speculative_selection(self, config):
        if self.fail_install:
            raise RuntimeError("injected selection publication failure")
        if not self._speculative_selection_installed:
            self.speculative_selection_config = config
            self._speculative_selection_installed = True
            return
        if self.speculative_selection_config == config:
            return
        raise RuntimeError(
            "speculative selection config is already installed"
        )
```

Add:

```python
def _activation_engine():
    return SimpleNamespace(
        scheduler=_ActivationScheduler(),
        model_runner=_model_runner(),
        speculative_runtime=None,
        speculative_runtime_poisoned=False,
        speculative_runtime_poison_reason=None,
    )


def test_atomic_activation_publishes_matching_runtime_and_selection():
    activate = _load_engine_method(
        "activate_speculative_runtime",
        {
            "LLMEngine": SimpleNamespace,
        },
    )
    runtime = EngineSpeculativeRuntime(
        draft_adapter=_Adapter(max_proposal_tokens=4),
    )
    engine = _activation_engine()
    engine.install_speculative_runtime = (
        lambda candidate: setattr(
            engine,
            "speculative_runtime",
            candidate,
        )
    )

    activate(engine, runtime)

    assert engine.speculative_runtime is runtime
    assert engine.scheduler.speculative_selection_config == (
        SpeculativeSelectionConfig(
            enabled=True,
            max_proposal_tokens=4,
        )
    )
    assert engine.scheduler._speculative_selection_installed


def test_atomic_activation_rolls_back_scheduler_on_runtime_failure():
    activate = _load_engine_method(
        "activate_speculative_runtime"
    )
    runtime = EngineSpeculativeRuntime(
        draft_adapter=_Adapter(max_proposal_tokens=4),
    )
    engine = _activation_engine()
    original = engine.scheduler.speculative_selection_config

    def fail_runtime_install(candidate):
        del candidate
        raise RuntimeError(
            "injected runtime publication failure"
        )

    engine.install_speculative_runtime = fail_runtime_install

    with pytest.raises(
        RuntimeError,
        match="runtime publication failure",
    ):
        activate(engine, runtime)

    assert engine.speculative_runtime is None
    assert engine.scheduler.speculative_selection_config == original
    assert not engine.scheduler._speculative_selection_installed
```

Also add tests that:

- identical runtime activation is idempotent;
- a different runtime fails before Scheduler mutation;
- an already-installed conflicting selection fails before runtime mutation;
- an invalid runtime leaves poison fields and both published states unchanged;
- injected Scheduler publication failure leaves all state unchanged.

- [ ] **Step 3: Run activation tests and verify RED**

Run:

```bash
python3 -m pytest \
  tools/test_engine_speculative_runtime.py \
  -k "activation" \
  -q
```

Expected: tests fail because `activate_speculative_runtime` is absent.

- [ ] **Step 4: Implement the atomic method**

Import the builder:

```python
from tinyvllm.engine.speculative_runtime import (
    EngineSpeculativeRuntime,
    build_engine_speculative_selection_config,
    validate_engine_speculative_runtime,
)
```

Add immediately before `install_speculative_runtime()`:

```python
def activate_speculative_runtime(
    self,
    runtime: EngineSpeculativeRuntime,
) -> None:
    candidate_selection = (
        build_engine_speculative_selection_config(
            runtime,
            model_runner=self.model_runner,
        )
    )
    current_runtime = self.speculative_runtime
    current_selection = (
        self.scheduler.speculative_selection_config
    )
    selection_installed = bool(
        self.scheduler._speculative_selection_installed
    )
    if current_runtime is runtime:
        if (
            selection_installed
            and current_selection == candidate_selection
        ):
            return
        raise RuntimeError(
            "speculative runtime and Scheduler selection "
            "are not atomically active"
        )
    if current_runtime is not None:
        raise RuntimeError(
            "speculative runtime is already installed"
        )
    if (
        selection_installed
        and current_selection != candidate_selection
    ):
        raise RuntimeError(
            "speculative selection config is already installed"
        )

    previous_runtime = current_runtime
    previous_poisoned = self.speculative_runtime_poisoned
    previous_poison_reason = (
        self.speculative_runtime_poison_reason
    )
    previous_selection = current_selection
    previous_selection_installed = selection_installed
    try:
        self.scheduler.install_speculative_selection(
            candidate_selection
        )
        self.install_speculative_runtime(runtime)
    except BaseException:
        self.scheduler.speculative_selection_config = (
            previous_selection
        )
        self.scheduler._speculative_selection_installed = (
            previous_selection_installed
        )
        self.speculative_runtime = previous_runtime
        self.speculative_runtime_poisoned = previous_poisoned
        self.speculative_runtime_poison_reason = (
            previous_poison_reason
        )
        raise
```

Do not add source-type or model-name conditionals.

- [ ] **Step 5: Run the complete runtime test file**

Run:

```bash
python3 -m pytest tools/test_engine_speculative_runtime.py -q
```

Expected: all tests pass, including existing install, lifecycle, rollback, and
mixed-accounting coverage.

- [ ] **Step 6: Run a generic source scan**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

paths = [
    Path("tinyvllm/engine/speculative_runtime.py"),
    Path("tinyvllm/engine/llm_engine.py"),
]
for path in paths:
    text = path.read_text()
    forbidden = ("Qwen", "qwen", "ngram", "sam_adapter")
    hits = [item for item in forbidden if item in text]
    if path.name == "llm_engine.py":
        hits = [
            item for item in hits
            if item not in ("Qwen", "qwen")
        ]
    if hits:
        raise SystemExit(f"{path}: forbidden activation branch markers {hits}")
print("generic activation source scan: PASS")
PY
```

Expected: `generic activation source scan: PASS`. The scan targets the new
activation/runtime region during implementation review; existing unrelated
Qwen3.5 engine code is not evidence of an activation branch.

- [ ] **Step 7: Review checkpoint**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/speculative_runtime.py \
  tinyvllm/engine/llm_engine.py
git diff --check
```

Expected: all commands succeed. Do not stage or commit.

---

### Task 3: Expose Direct Proposal and Target-Invocation Observations

**Files:**
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/test_engine_speculative_runtime.py`
- Modify: `tools/test_chunked_prefill.py`

**Interfaces:**
- Consumes: `PreparedNativeSpeculativeBatch.sequences`,
  `PreparedNativeSpeculativeSequence.proposal`,
  `first_target_callback_count`, `tail_callback_count`.
- Produces these `last_step_observation` fields:
  - `speculative_proposal_token_counts: dict[int, int]`
  - `speculative_proposal_row_count: int`
  - `speculative_first_target_callback_count: int`

- [ ] **Step 1: Add failing selected-path observation assertions**

In the existing selected-only engine runtime test, assert:

```python
assert engine.last_step_observation[
    "speculative_proposal_token_counts"
] == {selected_sequence_id: expected_proposal_length}
assert engine.last_step_observation[
    "speculative_proposal_row_count"
] == 1
assert engine.last_step_observation[
    "speculative_first_target_callback_count"
] == 1
```

In a no-proposal selected case, require the sequence entry to be present with
zero tokens and the proposal-row count to remain zero.

- [ ] **Step 2: Add failing ordinary-path default assertions**

Extend the ordinary chunked-prefill observation expectation:

```python
assert observation["speculative_proposal_token_counts"] == {}
assert observation["speculative_proposal_row_count"] == 0
assert observation["speculative_first_target_callback_count"] == 0
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
python3 -m pytest \
  tools/test_engine_speculative_runtime.py \
  tools/test_chunked_prefill.py \
  -k "observation or selected or default" \
  -q
```

Expected: failures identify the three absent fields.

- [ ] **Step 4: Populate the direct observations**

Initialize before selected execution:

```python
speculative_proposal_token_counts = {}
speculative_proposal_row_count = 0
speculative_first_target_callback_count = 0
```

After the authoritative KV and Scheduler commit:

```python
speculative_proposal_token_counts = {
    row.sequence_id: len(row.proposal.token_ids)
    for row in prepared_runtime.sequences
}
speculative_proposal_row_count = sum(
    1
    for count in speculative_proposal_token_counts.values()
    if count > 0
)
speculative_first_target_callback_count = (
    prepared_runtime.first_target_callback_count
)
```

Publish them in `last_step_observation` next to the existing speculative
counts:

```python
"speculative_proposal_token_counts": (
    speculative_proposal_token_counts
),
"speculative_proposal_row_count": (
    speculative_proposal_row_count
),
"speculative_first_target_callback_count": (
    speculative_first_target_callback_count
),
```

Do not infer proposals from accepted tokens or output lengths.

- [ ] **Step 5: Run focused and ordinary-path tests**

Run:

```bash
python3 -m pytest \
  tools/test_engine_speculative_runtime.py \
  tools/test_chunked_prefill.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 6: Review checkpoint**

Run:

```bash
git diff --check
```

Expected: success. Do not stage or commit.

---

### Task 4: Build a Dependency-Light Parity Artifact Contract and Verifier

**Files:**
- Create: `tools/speculative_tp1_parity_gate.py`
- Create: `tools/verify_speculative_tp1_parity_gate.py`
- Create: `tools/test_speculative_tp1_parity_gate.py`

**Interfaces:**
- Produces:
  - `aggregate_speculative_observations(observations: list[dict]) -> dict`
  - `build_parity_artifact(*, baseline: dict, speculative: dict, environment: dict, source_files: dict[str, str]) -> dict`
  - `validate_parity_artifact(payload: dict) -> dict`
  - verifier CLI: `python tools/verify_speculative_tp1_parity_gate.py --artifact PATH --repo-root PATH`

- [ ] **Step 1: Write failing counter aggregation tests**

Create fixtures with two speculative steps:

```python
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


def test_aggregate_uses_direct_production_observations():
    summary = gate.aggregate_speculative_observations(
        OBSERVATIONS
    )
    assert summary == {
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
```

Add rejection tests for negative counts, missing direct fields, boolean counts,
and accepted tokens greater than proposed tokens.

- [ ] **Step 2: Write failing artifact validation tests**

Use exact baseline/speculative token IDs and source hashes:

```python
def test_artifact_requires_exact_token_parity_and_real_selection():
    artifact = gate.build_parity_artifact(
        baseline={
            "outputs": [[10, 11, 12]],
            "prompt_token_ids": [[1, 2, 3]],
        },
        speculative={
            "outputs": [[10, 11, 12]],
            "prompt_token_ids": [[1, 2, 3]],
            "summary": {
                "selected_rows": 1,
                "proposal_rows": 1,
                "proposed_tokens": 2,
                "accepted_draft_tokens": 1,
                "first_target_callbacks": 1,
                "tail_callbacks": 1,
                "target_invocations": 2,
                "acceptance_rate": 0.5,
                "accepted_tokens_per_target_invocation": 0.5,
            },
        },
        environment=_environment_fixture(),
        source_files={"tinyvllm/engine/llm_engine.py": "a" * 64},
    )
    assert gate.validate_parity_artifact(artifact)[
        "status"
    ] == "PASS"
```

Add tamper tests for:

- token divergence;
- zero selected rows;
- zero proposal rows;
- zero first-target callbacks;
- zero tail callbacks;
- TP size other than 1;
- nonzero temperature;
- missing model/tokenizer/source identity;
- malformed SHA-256;
- a claim that performance improved without controlled timing evidence.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
python3 -m pytest tools/test_speculative_tp1_parity_gate.py -q
```

Expected: import fails because the gate modules do not exist.

- [ ] **Step 4: Implement strict pure-data validation**

Use schema version `1` and require this top-level shape:

```python
{
    "schema_version": 1,
    "status": "PASS",
    "claim_scope": (
        "Qwen3-0.6B BF16 TP1 greedy exact-token parity "
        "for atomic n-gram speculative activation"
    ),
    "baseline": {...},
    "speculative": {...},
    "environment": {
        "model_path": str,
        "model_identifier": str,
        "tokenizer_identifier": str,
        "tensor_parallel_size": 1,
        "temperature": 0.0,
        "dtype": str,
        "device_name": str,
        "python_version": str,
        "torch_version": str,
        "command": list[str],
    },
    "source_files": {
        "relative/path.py": "64-character lowercase sha256",
    },
    "limitations": [
        "no TPOT or throughput improvement claim",
        "no TP4 claim",
        "no long-context claim",
        "no offload reduction claim",
        "no learned-drafter or MTP claim",
    ],
}
```

`validate_parity_artifact()` must recompute parity from the stored output token
lists and reject any artifact whose `status` says `PASS` while a required gate
is absent.

- [ ] **Step 5: Implement independent source verification**

In `verify_speculative_tp1_parity_gate.py`, load the artifact, call the pure
validator, then recompute every listed SHA-256 relative to `--repo-root`:

```python
for relative_path, expected_sha256 in source_files.items():
    path = repo_root / relative_path
    actual_sha256 = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    if actual_sha256 != expected_sha256:
        raise VerificationError(
            f"source hash mismatch: {relative_path}"
        )
```

Print a deterministic JSON summary containing `status`, `schema_version`,
`output_sequences`, `proposed_tokens`, `accepted_draft_tokens`, and
`target_invocations`.

- [ ] **Step 6: Run artifact and tamper tests**

Run:

```bash
python3 -m pytest tools/test_speculative_tp1_parity_gate.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Review checkpoint**

Run:

```bash
python3 -m py_compile \
  tools/speculative_tp1_parity_gate.py \
  tools/verify_speculative_tp1_parity_gate.py
git diff --check
```

Expected: success. Do not stage or commit.

---

### Task 5: Add the Real Loaded-Model TP1 Runner

**Files:**
- Modify: `tools/speculative_tp1_parity_gate.py`
- Test: `tools/test_speculative_tp1_parity_gate.py`

**Interfaces:**
- Consumes:
  - `tinyvllm.LLM`
  - `SamplingParams(temperature=0.0, max_tokens=N, ignore_eos=False)`
  - `EngineSpeculativeRuntime`
  - `NGramDraftAdapter`
  - `LLM.activate_speculative_runtime(runtime)`
- Produces CLI:
  - `python tools/speculative_tp1_parity_gate.py run --model PATH --out PATH`

- [ ] **Step 1: Add failing runner tests with a fake engine factory**

Define a fake engine that records activation, requests, outputs, and step
observations. Test:

```python
def test_run_case_activates_only_speculative_engine():
    baseline_factory = _FakeEngineFactory(
        outputs=[[21, 22, 23]],
        observations=[],
    )
    speculative_factory = _FakeEngineFactory(
        outputs=[[21, 22, 23]],
        observations=OBSERVATIONS,
    )

    baseline = gate.run_engine_case(
        engine_factory=baseline_factory,
        model_path="/models/Qwen3-0.6B",
        prompts=gate.DEFAULT_PROMPTS,
        max_tokens=16,
        activate=False,
        ngram_size=3,
        max_proposal_tokens=4,
    )
    speculative = gate.run_engine_case(
        engine_factory=speculative_factory,
        model_path="/models/Qwen3-0.6B",
        prompts=gate.DEFAULT_PROMPTS,
        max_tokens=16,
        activate=True,
        ngram_size=3,
        max_proposal_tokens=4,
    )

    assert baseline_factory.activations == []
    assert len(speculative_factory.activations) == 1
    assert baseline["outputs"] == speculative["outputs"]
    assert speculative["summary"]["proposal_rows"] > 0
```

Add a failure test where outputs match but the fake speculative engine produces
no selected/proposal/verifier observations.

- [ ] **Step 2: Run runner tests and verify RED**

Run:

```bash
python3 -m pytest \
  tools/test_speculative_tp1_parity_gate.py \
  -k "run_case" \
  -q
```

Expected: `run_engine_case` is absent.

- [ ] **Step 3: Implement isolated baseline/speculative execution**

Use a default prompt bank with repetition sufficient for n-gram proposals:

```python
DEFAULT_PROMPTS = (
    "alpha beta gamma alpha beta gamma alpha beta gamma",
    (
        "The sky is blue. The grass is green. "
        "The sky is blue. The grass is green. "
        "Continue the pattern:"
    ),
)
```

For each case:

1. instantiate `LLM(model_path, tensor_parallel_size=1, enforce_eager=True,
   max_model_len=4096, max_num_seqs=4)`;
2. if `activate=True`, construct
   `NGramDraftAdapter(ngram_size=3, max_proposal_tokens=4)` and call
   `activate_speculative_runtime(EngineSpeculativeRuntime(adapter))`;
3. tokenize and store each prompt;
4. add each request with greedy `SamplingParams`;
5. drive `step()` until `is_finished()`;
6. copy every non-`None` `last_step_observation`;
7. sort final outputs by sequence ID;
8. call `engine.exit()` in `finally`;
9. return outputs, prompt IDs, observations, direct aggregate summary, and
   elapsed wall time as diagnostic-only data.

The artifact validator must not interpret elapsed wall time as a performance
improvement.

- [ ] **Step 4: Implement source and environment capture**

Hash this exact source set:

```python
SOURCE_FILES = (
    "tinyvllm/engine/block_manager.py",
    "tinyvllm/engine/llm_engine.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tinyvllm/engine/speculative_execution.py",
    "tinyvllm/engine/speculative_model_runner.py",
    "tinyvllm/engine/speculative_runtime.py",
    "tinyvllm/engine/speculative_selection.py",
    "tinyvllm/speculative/adapter.py",
    "tinyvllm/speculative/batch_runtime.py",
    "tinyvllm/speculative/ngram.py",
    "tinyvllm/speculative/ngram_adapter.py",
    "tinyvllm/speculative/runtime.py",
    "tools/speculative_tp1_parity_gate.py",
    "tools/verify_speculative_tp1_parity_gate.py",
)
```

Record `torch.cuda.get_device_name(0)`, Torch/Python versions, dtype from the
engine config, tokenizer `name_or_path`, exact CLI argv, and resolved model
path.

- [ ] **Step 5: Implement atomic artifact writing**

Reject an existing output path. Write to a sibling temporary file, flush, and
replace:

```python
temporary = output_path.with_suffix(
    output_path.suffix + ".tmp"
)
temporary.write_text(
    json.dumps(
        artifact,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)
temporary.replace(output_path)
```

Call `validate_parity_artifact()` before writing.

- [ ] **Step 6: Run mocked runner and full contract tests**

Run:

```bash
python3 -m pytest tools/test_speculative_tp1_parity_gate.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Review checkpoint**

Run:

```bash
python3 -m py_compile tools/speculative_tp1_parity_gate.py
git diff --check
```

Expected: success. Do not stage or commit.

---

### Task 6: Package the Remote TP1 Execution

**Files:**
- Create: `tools/run_speculative_tp1_parity_gate_remote.sh`
- Modify: `tools/test_speculative_tp1_parity_gate.py`

**Interfaces:**
- Produces a local artifact under:
  - `artifacts/speculative_tp1_parity/YYYYMMDDTHHMMSSZ/result.json`
  - `artifacts/speculative_tp1_parity/YYYYMMDDTHHMMSSZ/verify.json`
  - `artifacts/speculative_tp1_parity/YYYYMMDDTHHMMSSZ/remote.log`

- [ ] **Step 1: Add a source-contract test for the remote script**

Require the script to contain:

```python
required = (
    "sitian@10.232.195.203",
    "/tmp/ssh-sitian-10.232.195.203",
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python",
    "/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B",
    "CUDA_VISIBLE_DEVICES=",
    "speculative_tp1_parity_gate.py",
    "verify_speculative_tp1_parity_gate.py",
)
for value in required:
    assert value in script_text
```

Also assert the script does not contain `git checkout`, `git reset`, `git
clean`, `git stash`, `git add`, `git commit`, or `git push`.

- [ ] **Step 2: Run the source-contract test and verify RED**

Run:

```bash
python3 -m pytest \
  tools/test_speculative_tp1_parity_gate.py \
  -k "remote_script" \
  -q
```

Expected: failure because the script does not exist.

- [ ] **Step 3: Implement the remote wrapper**

Use:

```bash
#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
CONTROL_SOCKET="${CONTROL_SOCKET:-/tmp/ssh-sitian-10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B}"
REMOTE_REPO="${REMOTE_REPO:-/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge}"
GPU_ID="${GPU_ID:-0}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOCAL_OUT="artifacts/speculative_tp1_parity/${RUN_TAG}"
REMOTE_OUT="/tmp/tinyllmforge-speculative-tp1-${RUN_TAG}"
```

Verify the existing ControlMaster:

```bash
ssh -S "${CONTROL_SOCKET}" -o BatchMode=yes \
  "${REMOTE_HOST}" true
```

Sync the exact `SOURCE_FILES` set plus the runner, verifier, and required
package `__init__.py` files using `rsync --relative`. Execute:

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTHONPATH="${REMOTE_REPO}" \
"${REMOTE_PYTHON}" \
  "${REMOTE_REPO}/tools/speculative_tp1_parity_gate.py" run \
  --model "${MODEL_PATH}" \
  --max-tokens 32 \
  --ngram-size 3 \
  --max-proposal-tokens 4 \
  --out "${REMOTE_OUT}/result.json"
```

Then execute the independent verifier remotely, download all three files, and
run the verifier again locally against the synchronized local source tree.

- [ ] **Step 4: Run source-contract tests**

Run:

```bash
python3 -m pytest tools/test_speculative_tp1_parity_gate.py -q
bash -n tools/run_speculative_tp1_parity_gate_remote.sh
```

Expected: all tests pass and Bash syntax is valid.

- [ ] **Step 5: Review checkpoint**

Run:

```bash
git diff --check
```

Expected: success. Do not stage or commit.

---

### Task 7: Run the Local Regression Matrix

**Files:**
- No production edits unless a regression exposes a defect in the owned scope.

**Interfaces:**
- Validates all interfaces produced by Tasks 1-6.

- [ ] **Step 1: Run the focused activation and artifact tests**

Run:

```bash
python3 -m pytest \
  tools/test_engine_speculative_runtime.py \
  tools/test_speculative_tp1_parity_gate.py \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_engine_speculative_execution.py \
  tools/test_llm_engine_speculative_selection_source.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 2: Run the established speculative regression matrix**

Run the same file set used for the recorded `322 passed` matrix:

```bash
python3 -m pytest \
  tools/test_speculative_adapter.py \
  tools/test_speculative_source_adapters.py \
  tools/test_speculative_runtime.py \
  tools/test_speculative_batch_runtime.py \
  tools/test_speculative_kv_transaction.py \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_engine_speculative_execution.py \
  tools/test_scheduler_speculative_selection.py \
  tools/test_speculative_selection_record.py \
  -q
```

Expected: all tests pass. Record the exact count from stdout rather than
assuming it remains `322`.

- [ ] **Step 3: Run Torch-dependent and Scheduler regressions**

Run:

```bash
/opt/homebrew/bin/python3.12 tools/test_chunked_prefill.py
/opt/homebrew/bin/python3.12 tools/test_native_verifier_attention.py
/opt/homebrew/bin/python3.12 tools/test_scheduler_prefill_commit_hook.py
```

Expected:

- chunked prefill passes;
- native verifier attention passes, with CUDA numerical cases explicitly
  deferred if no local CUDA exists;
- Scheduler prefill hook passes.

- [ ] **Step 4: Run compatibility and hygiene checks**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/speculative_runtime.py \
  tinyvllm/engine/llm_engine.py \
  tools/speculative_tp1_parity_gate.py \
  tools/verify_speculative_tp1_parity_gate.py
/opt/homebrew/bin/python3.12 -m py_compile \
  tinyvllm/engine/speculative_runtime.py \
  tinyvllm/engine/llm_engine.py \
  tools/speculative_tp1_parity_gate.py \
  tools/verify_speculative_tp1_parity_gate.py
git diff --check
test -z "$(git diff --cached --name-only)"
```

Expected: every command succeeds and staged diff is empty.

---

### Task 8: Execute and Independently Verify the Real TP1 Gate

**Files:**
- Generated only: `artifacts/speculative_tp1_parity/YYYYMMDDTHHMMSSZ/...`

**Interfaces:**
- Consumes the remote wrapper from Task 6.
- Produces one source-bound PASS or an explicit incomplete/failure artifact.

- [ ] **Step 1: Verify SSH transport without starting GPU work**

Run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
ssh -S /tmp/ssh-sitian-10.232.195.203 \
  -o BatchMode=yes \
  sitian@10.232.195.203 \
  'echo remote-ok'
```

Expected: `remote-ok`. If it fails, inspect `klist`, the ControlMaster socket,
and `jump-proxy-hl` routing before any GPU claim.

- [ ] **Step 2: Run the packaged gate**

Run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
GPU_ID=0 \
bash tools/run_speculative_tp1_parity_gate_remote.sh
```

Expected: the script prints the local artifact directory and exits zero.

- [ ] **Step 3: Inspect the raw result**

Run:

```bash
RUN_DIR="$(ls -dt artifacts/speculative_tp1_parity/* | head -1)"
python3 - <<'PY' "${RUN_DIR}/result.json"
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
print(json.dumps({
    "status": payload["status"],
    "model": payload["environment"]["model_identifier"],
    "tp": payload["environment"]["tensor_parallel_size"],
    "temperature": payload["environment"]["temperature"],
    "outputs_equal": (
        payload["baseline"]["outputs"]
        == payload["speculative"]["outputs"]
    ),
    "summary": payload["speculative"]["summary"],
}, indent=2, sort_keys=True))
PY
```

Expected:

- `status == "PASS"`;
- `tp == 1`;
- `temperature == 0.0`;
- `outputs_equal == true`;
- `selected_rows > 0`;
- `proposal_rows > 0`;
- `proposed_tokens > 0`;
- `first_target_callbacks > 0`;
- `tail_callbacks > 0`.

- [ ] **Step 4: Run independent local verification**

Run:

```bash
RUN_DIR="$(ls -dt artifacts/speculative_tp1_parity/* | head -1)"
python3 tools/verify_speculative_tp1_parity_gate.py \
  --artifact "${RUN_DIR}/result.json" \
  --repo-root .
```

Expected: JSON with `"status": "PASS"` and the same direct counters as the raw
artifact.

- [ ] **Step 5: Preserve a failure boundary if the gate cannot pass**

If model loading, SSH, CUDA, or runtime execution fails, retain `remote.log`
and write an `INCOMPLETE` note containing the exact command, exception, host,
model path, and missing gate. Do not convert environment failure into PASS and
do not claim parity from local unit tests.

---

### Task 9: Update Audit and Handoff with Actual Evidence

**Files:**
- Modify: `docs/superpowers/audits/2026-08-12-generic-inference-optimization-goal-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes exact local test output and the independently verified TP1 artifact.
- Produces an updated prompt-to-artifact checklist and next critical path.

- [ ] **Step 1: Update activation status**

Record:

- atomic activation API path and method name;
- all-or-nothing publication and rollback tests;
- exact proposal-limit derivation;
- idempotence and conflict behavior;
- default-off and fail-closed boundaries.

- [ ] **Step 2: Update TP1 parity evidence**

Record:

- artifact path;
- independent verifier command and result;
- model/checkpoint/tokenizer identity;
- TP size, dtype, GPU, prompt count, output budget;
- exact output parity;
- proposed and accepted token counts;
- first-target, tail, and total target invocation counts.

- [ ] **Step 3: Keep the classification honest**

The audit must remain `NOT_PROMOTABLE`. Explicitly retain these missing gates:

- second model structure;
- TP4;
- 4K/16K/32K+;
- batch 1/4/mixed promotion campaign;
- controlled TPOT, TTFT, throughput, and memory measurements;
- real speculative/offload H2D/D2H integration;
- learned drafter;
- MTP adapter.

- [ ] **Step 4: Set the next critical path**

The next design gate becomes speculative transaction integration with
`KVOffloadMVP0` residency:

- reserve/materialize creates or protects residency state;
- accepted commit transfers ownership exactly once;
- rejected rollback invalidates speculative residency exactly once;
- counters come only from real `KVOffloadMVP0` H2D/D2H events.

- [ ] **Step 5: Final hygiene checkpoint**

Run:

```bash
rg -n "NOT_PROMOTABLE|atomic activation|TP1|KVOffloadMVP0" \
  docs/superpowers/audits/2026-08-12-generic-inference-optimization-goal-audit.md \
  AGENT_HANDOFF_STATE.md
git diff --check
test -z "$(git diff --cached --name-only)"
```

Expected: the updated state and remaining boundaries are present, diff hygiene
passes, and staged diff remains empty. Do not stage or commit.
