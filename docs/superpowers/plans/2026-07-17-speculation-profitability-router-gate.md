# Speculation Profitability Router Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bypass unprofitable one-token speculation, retain the proven
zero-replay native verifier for compatible multi-token drafts, and produce
source-auditable controlled and real-source gates that compare routed
speculation with normal decode end to end.

**Architecture:** Add a pure deterministic router contract under
`tinyvllm/speculative/`, then add one profiler wrapper that either records a
baseline fallback or delegates to the existing
`verify_and_commit_block()` with `verifier_mode="native"`. Build a separate
`speculation_router_gate.py` rather than expanding the completed native
verifier gate. Reuse a shared source-audit module for immutable local,
remote, and downloaded source identity. The controlled stage measures a
target-derived acceptance envelope and can classify only
`READY_FOR_REAL_DRAFTER_GATE`; the real-source stage includes all draft
construction cost and is the only stage that may classify `GO`.

**Tech Stack:** Python 3 standard library and dataclasses, existing
TinyLLMForge profiler/native verifier/oracle, PyTorch and FlashAttention on
the remote host, dependency-light Python test scripts, Bash, Git, SSH/SCP,
Qwen3-0.6B.

## Global Constraints

- The normative design is
  `docs/superpowers/specs/2026-07-17-speculation-profitability-router-gate-design.md`.
- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`; do not
  modify `/Users/bytedance/dev/TinyLLMForge`.
- Execute inline in the current session; do not dispatch subagents.
- Preserve the completed native verifier implementation and its oracle
  tolerances. Do not add a second verifier.
- `draft_len <= 1` must use normal decode without speculative block
  reservation, `prepare_spec_verify()`, a `spec_verify` forward, accepted-KV
  replay/copy/rematerialization, or a draft-token override.
- Compatible `draft_len >= 2` uses the existing native verifier.
- Controlled-envelope incompatibility fails before KV mutation. Real-source
  incompatibility records `baseline_incompatible` and falls back to normal
  decode.
- Router decisions use only information available before target
  verification. They must not inspect acceptance, target logits, or future
  timing.
- The first router is fixed and stateless. Do not add adaptive thresholds,
  prompt exceptions, confidence tuning, or an online EMA.
- Controlled target-derived drafts are correctness/performance-envelope
  fixtures only. They may classify only `READY_FOR_REAL_DRAFTER_GATE`,
  `NO_GO`, or `INCOMPLETE`.
- Only a named, source-attributed, non-target-derived real source may produce
  a performance `GO`.
- Existing prompt lookup, n-gram, adaptive n-gram, and SAM sources are
  negative controls because their canonical gates are already `NO_GO`.
  Do not tune them on the same prompt bank and present the result as a new
  source.
- Controlled thresholds are fixed at a routed-versus-baseline median elapsed
  ratio below `0.95` in at least one accepted `K>=2` region, with no required
  lifecycle case above `1.05`.
- Real-source thresholds are fixed at at least `5%` aggregate median elapsed
  improvement, at least `5%` aggregate tokens/s improvement, natural and
  transition-heavy median elapsed ratios at most `1.00`, and no required
  prompt regression above `10%`.
- Exact greedy token, acceptance, lifecycle, KV, logit, and 16-token
  continuation equality remain hard gates.
- Every native event must report zero accepted-KV replay, copy, and
  rematerialization.
- Memory measurements are diagnostic only and never satisfy a speed or
  memory-reduction claim.
- GPU/model experiments run only on `sitian@10.232.195.203` with Python
  `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python` and model
  `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`.
- Every model process receives unique dynamic `TINYVLLM_DIST_PORT` and
  `MASTER_PORT` values.
- Use run-local remote directories and `TMPDIR`. Do not kill unrelated
  processes, clear shared `/tmp`, or mutate the remote checkout.
- The existing K1 candidate in `tools/profile_ngram_commit.py` and
  `tools/test_ngram_speculative.py` is a completed `NO_GO`. Preserve its exact
  canonical patch in the existing artifact, verify that identity, then
  restore those two worktree files to committed `HEAD` before router
  implementation. Never commit the failed K1 patch.
- Keep all existing adaptive n-gram artifacts. Do not delete them to obtain a
  clean status.
- Commit implementation in selective commits; never use `git add -A` while
  experiment artifacts are untracked.
- A controlled `READY_FOR_REAL_DRAFTER_GATE` is not a product `GO`.
- Do not begin learned-drafter checkpoint integration in this plan. If the
  controlled gate is ready, write a separate design for the smallest
  attributable EAGLE-style or smaller-model source.

## File Structure

- Create `tinyvllm/speculative/router.py`: immutable router decision type,
  fixed decision ordering, route constants, and pure decision function.
- Create `tools/test_speculation_router.py`: dependency-light pure contract
  and profiler-wrapper tests.
- Modify `tools/profile_ngram_commit.py`: profiler-owned
  `route_and_verify_draft()` wrapper, baseline fallback event, route
  instrumentation, candidate-loop integration, and result aggregation.
- Modify `tools/test_ngram_speculative.py`: retain native primitive tests and
  add routed lifecycle/instrumentation regressions.
- Create `tools/source_audit.py`: reusable owned-source expansion, hashing,
  snapshot, patch reconstruction, and remote-preflight validation.
- Modify `tools/adaptive_ngram_gate.py`: import and re-export shared
  source-audit functions without changing its artifact schema or canonical
  classification.
- Modify `tools/test_adaptive_ngram_gate.py`: prove the extraction preserves
  existing source-audit behavior.
- Create `tools/speculation_router_gate.py`: controlled/real manifests, case
  matrices, process runner, resume reconciliation, classification, report,
  hashes, and artifact verification.
- Create `tools/test_speculation_router_gate.py`: complete synthetic
  classification, source identity, resume, port, report, and tamper tests.
- Modify `tools/native_verifier_oracle.py`: add `routed_native` policy and
  route evidence while retaining existing policies and comparisons.
- Modify `tools/test_native_verifier_oracle.py`: routed fallback/native
  payload and comparison tests.
- Create `tools/run_speculation_router_gate_remote.sh`: immutable staged
  source, remote preflight/capability, controlled and real-source modes,
  detached canonical execution, artifact download, and local verification.
- Create `tools/test_run_speculation_router_gate_remote.py`: shell contract,
  owned-source, dynamic-port, run-local path, and canonical-file tests.
- Modify `README.md`: commands, classifications, measured results, and claim
  boundaries after real remote evidence exists.
- Modify `AGENT_HANDOFF_STATE.md`: exact source identity, commands, artifact
  paths, result, limitations, and next direction after real evidence exists.
- Generate artifacts only under
  `experiments/speculation_router/<run_tag>/`.

---

### Task 1: Retire the Failed K1 Dirty Candidate Without Losing Evidence

**Files:**
- Verify:
  `experiments/adaptive_ngram/20260717-k1-sam-canonical/source.patch`
- Verify:
  `experiments/adaptive_ngram/20260717-k1-sam-canonical/source_evidence.json`
- Restore from `HEAD`: `tools/profile_ngram_commit.py`
- Restore from `HEAD`: `tools/test_ngram_speculative.py`

**Interfaces:**
- Consumes: canonical K1 source patch and tree identity already recorded by
  the adaptive gate.
- Produces: a router-development worktree where the two shared source files
  contain committed native-verifier code only.

- [ ] **Step 1: Verify the current dirty patch is the canonical K1 patch**

Run:

```bash
python3 - <<'PY'
import hashlib
import json
import subprocess
from pathlib import Path

root = Path(".").resolve()
artifact = root / "experiments/adaptive_ngram/20260717-k1-sam-canonical"
evidence = json.loads((artifact / "source_evidence.json").read_text())
recorded = evidence["patch_sha256"]
artifact_patch = hashlib.sha256(
    (artifact / "source.patch").read_bytes()
).hexdigest()
current_patch = subprocess.run(
    [
        "git", "diff", "--binary", "--no-ext-diff",
        evidence["base_commit"], "--",
        "tools/profile_ngram_commit.py",
        "tools/test_ngram_speculative.py",
    ],
    cwd=root,
    check=True,
    capture_output=True,
).stdout
current = hashlib.sha256(current_patch).hexdigest()
assert artifact_patch == recorded
assert current == recorded, (current, recorded)
print("K1_PATCH_IDENTITY_OK", recorded)
PY
```

Expected: `K1_PATCH_IDENTITY_OK
2c23549c6e8e875cab0d1b6dc9b79031a22bacdf1da3aeb3ef20a825cbb13392`.

- [ ] **Step 2: Restore only the two failed K1 candidate files**

Run:

```bash
git restore --source=HEAD -- \
  tools/profile_ngram_commit.py \
  tools/test_ngram_speculative.py
```

Expected: command exits `0`; all adaptive experiment directories remain.

- [ ] **Step 3: Verify the retirement boundary**

Run:

```bash
git status --short
git diff -- tools/profile_ngram_commit.py tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_oracle.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_gate.py
```

Expected:

- no diff for the two restored files;
- the three adaptive artifact directories remain untracked;
- all three dependency-light suites pass.

No commit is required. The failed candidate is preserved by its immutable
artifact and intentionally absent from Git history.

---

### Task 2: Pure Fixed Router Contract

**Files:**
- Create: `tinyvllm/speculative/router.py`
- Create: `tools/test_speculation_router.py`

**Interfaces:**
- Produces:
  `RouteName = Literal["baseline_short_draft", "baseline_finished", "baseline_output_budget", "baseline_incompatible", "native_multi_token"]`
- Produces:
  `SpeculationRoute(name: RouteName, draft_len: int, native_compatible: bool, fallback_reason: str | None)`
- Produces:
  `choose_speculation_route(*, draft_len: int, finished: bool, remaining_output_budget: int, native_compatible: bool, compatibility_reason: str | None = None, allow_incompatible_fallback: bool = False) -> SpeculationRoute`
- Produces:
  `route_to_dict(route: SpeculationRoute) -> dict[str, object]`

- [ ] **Step 1: Write failing router contract tests**

Create `tools/test_speculation_router.py`:

```python
from tinyvllm.speculative.router import (
    choose_speculation_route,
    route_to_dict,
)


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
```

Add a direct `main()` that calls every test and prints
`speculation router tests passed`.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_speculation_router.py
```

Expected: import failure for `tinyvllm.speculative.router`.

- [ ] **Step 3: Implement the immutable router**

Create `tinyvllm/speculative/router.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


RouteName = Literal[
    "baseline_short_draft",
    "baseline_finished",
    "baseline_output_budget",
    "baseline_incompatible",
    "native_multi_token",
]


@dataclass(frozen=True)
class SpeculationRoute:
    name: RouteName
    draft_len: int
    native_compatible: bool
    fallback_reason: str | None = None


def choose_speculation_route(
    *,
    draft_len: int,
    finished: bool,
    remaining_output_budget: int,
    native_compatible: bool,
    compatibility_reason: str | None = None,
    allow_incompatible_fallback: bool = False,
) -> SpeculationRoute:
    draft_len = int(draft_len)
    remaining_output_budget = int(remaining_output_budget)
    if draft_len < 0:
        raise ValueError("draft_len must be >= 0")
    if finished:
        return SpeculationRoute(
            "baseline_finished", draft_len, bool(native_compatible)
        )
    if remaining_output_budget <= 0:
        return SpeculationRoute(
            "baseline_output_budget", draft_len, bool(native_compatible)
        )
    if draft_len <= 1:
        return SpeculationRoute(
            "baseline_short_draft", draft_len, bool(native_compatible)
        )
    if not native_compatible:
        reason = compatibility_reason or "native verifier incompatible"
        if not allow_incompatible_fallback:
            raise ValueError(reason)
        return SpeculationRoute(
            "baseline_incompatible", draft_len, False, reason
        )
    return SpeculationRoute("native_multi_token", draft_len, True)


def route_to_dict(route: SpeculationRoute) -> dict[str, object]:
    return asdict(route)
```

- [ ] **Step 4: Run focused and existing contract tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_speculation_router.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_contract.py
python3 -m py_compile tinyvllm/speculative/router.py
```

Expected: both suites pass and compilation exits `0`.

- [ ] **Step 5: Commit the router contract**

```bash
git add \
  tinyvllm/speculative/router.py \
  tools/test_speculation_router.py
git commit -m "feat: add fixed speculation profitability router"
```

---

### Task 3: Profiler Route-and-Verify Wrapper

**Files:**
- Modify: `tools/profile_ngram_commit.py`
- Modify: `tools/test_speculation_router.py`
- Modify: `tools/test_ngram_speculative.py`

**Interfaces:**
- Consumes:
  `choose_speculation_route(*, draft_len: int, finished: bool, remaining_output_budget: int, native_compatible: bool, compatibility_reason: str | None = None, allow_incompatible_fallback: bool = False) -> SpeculationRoute`
- Produces:
  `route_and_verify_draft(llm, seq, draft_tokens: list[int], *, draft_source: str, allow_incompatible_fallback: bool, **verify_kwargs) -> dict`
- Produces baseline fallback event fields:
  `route`, `route_fallback_reason`, `verifier_mode`, `draft_len`,
  `draft_tokens`, `accepted_tokens`, `accepted_count`,
  `accepted_kv_rematerialization`, `accepted_kv_copy_calls`,
  `accepted_kv_replay_calls`, `target_forward_count`, `timing_ms`,
  `speculative_reservation_attempted`, `spec_verify_prepare_calls`,
  `spec_verify_forward_calls`.

- [ ] **Step 1: Add failing wrapper tests with mutation counters**

Extend `tools/test_speculation_router.py` with fakes:

```python
class FakeBlockManager:
    def __init__(self):
        self.reserve_calls = 0

    def reserve_append_blocks(self, seq, count):
        self.reserve_calls += 1
        raise AssertionError("short draft must not reserve blocks")


class FakeRunner:
    def __init__(self, compatible=True):
        self.compatible = compatible
        self.validate_calls = 0
        self.prepare_calls = 0
        self.run_model_calls = 0

    def _validate_spec_verify_compatibility(self, **kwargs):
        self.validate_calls += 1
        if not self.compatible:
            raise RuntimeError("kv_offload_mvp0 is unsupported")


class FakeSeq:
    is_finished = False
    max_tokens = 32
    num_completion_tokens = 4


def test_short_draft_wrapper_performs_no_target_or_kv_work():
    llm = type("LLM", (), {})()
    llm.model_runner = FakeRunner()
    llm.scheduler = type("Scheduler", (), {})()
    llm.scheduler.block_manager = FakeBlockManager()
    event = profile.route_and_verify_draft(
        llm,
        FakeSeq(),
        [7],
        draft_source="fixture",
        allow_incompatible_fallback=False,
    )
    assert event["route"] == "baseline_short_draft"
    assert event["accepted_count"] == 0
    assert event["target_forward_count"] == 0
    assert event["speculative_reservation_attempted"] is False
    assert event["spec_verify_prepare_calls"] == 0
    assert event["spec_verify_forward_calls"] == 0
    assert llm.scheduler.block_manager.reserve_calls == 0


def test_real_source_incompatibility_falls_back_before_mutation():
    llm = type("LLM", (), {})()
    llm.model_runner = FakeRunner(compatible=False)
    llm.scheduler = type("Scheduler", (), {})()
    llm.scheduler.block_manager = FakeBlockManager()
    event = profile.route_and_verify_draft(
        llm,
        FakeSeq(),
        [1, 2, 3, 4],
        draft_source="fixture",
        allow_incompatible_fallback=True,
    )
    assert event["route"] == "baseline_incompatible"
    assert "kv_offload_mvp0" in event["route_fallback_reason"]
    assert llm.scheduler.block_manager.reserve_calls == 0
```

Add a delegation test by temporarily replacing
`profile.verify_and_commit_block` with a fake and asserting `K=4` passes
`verifier_mode="native"` and returns `route="native_multi_token"`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_speculation_router.py
```

Expected: failure because `route_and_verify_draft` is absent.

- [ ] **Step 3: Implement baseline event and wrapper**

In `tools/profile_ngram_commit.py`, import the router and add:

```python
def _baseline_route_event(
    *,
    route,
    draft_tokens: list[int],
    draft_source: str,
) -> dict:
    return {
        "route": route.name,
        "route_fallback_reason": route.fallback_reason,
        "verifier_mode": "baseline",
        "draft_source": draft_source,
        "draft_len": len(draft_tokens),
        "draft_tokens": list(draft_tokens),
        "target_tokens": [],
        "accepted_tokens": [],
        "accepted_count": 0,
        "accepted_kv_rematerialization": {
            "rematerialized_tokens": [],
            "decode_calls": 0,
            "elapsed_ms": 0.0,
        },
        "accepted_kv_copy_calls": 0,
        "accepted_kv_replay_calls": 0,
        "target_forward_count": 0,
        "speculative_reservation_attempted": False,
        "spec_verify_prepare_calls": 0,
        "spec_verify_forward_calls": 0,
        "timing_ms": {"verify_commit_total_ms": 0.0},
        "finished": False,
    }


def route_and_verify_draft(
    llm,
    seq,
    draft_tokens: list[int],
    *,
    draft_source: str,
    allow_incompatible_fallback: bool,
    **verify_kwargs,
) -> dict:
    compatibility_reason = None
    native_compatible = True
    if len(draft_tokens) >= 2:
        try:
            llm.model_runner._validate_spec_verify_compatibility(
                seq_count=1,
                linear_draft=True,
                greedy=True,
                mixed_batch=False,
            )
        except Exception as exc:
            native_compatible = False
            compatibility_reason = str(exc)
    route = choose_speculation_route(
        draft_len=len(draft_tokens),
        finished=bool(seq.is_finished),
        remaining_output_budget=max(
            0, seq.max_tokens - seq.num_completion_tokens
        ),
        native_compatible=native_compatible,
        compatibility_reason=compatibility_reason,
        allow_incompatible_fallback=allow_incompatible_fallback,
    )
    if route.name != "native_multi_token":
        return _baseline_route_event(
            route=route,
            draft_tokens=draft_tokens,
            draft_source=draft_source,
        )
    event = verify_and_commit_block(
        llm,
        seq,
        draft_tokens,
        draft_source=draft_source,
        verifier_mode="native",
        **verify_kwargs,
    )
    event.update({
        "route": route.name,
        "route_fallback_reason": None,
        "speculative_reservation_attempted": True,
        "spec_verify_prepare_calls": int(event["query_len"] > 0),
        "spec_verify_forward_calls": int(event["query_len"] > 0),
    })
    return event
```

Do not change `verify_and_commit_block()` semantics.

- [ ] **Step 4: Replace candidate-loop direct native calls**

In `run_candidate_only_profile()` and the paired profiler path, call
`route_and_verify_draft()` when a new CLI option
`--speculation-routing fixed-profitability` is selected. Preserve
`--speculation-routing always-native` for the controlled comparator and
`--speculation-routing disabled` for legacy behavior.

Add parser choices:

```python
parser.add_argument(
    "--speculation-routing",
    choices=("disabled", "always-native", "fixed-profitability"),
    default="disabled",
)
parser.add_argument(
    "--allow-incompatible-fallback",
    action="store_true",
)
```

For `fixed-profitability`, every proposal including `K=0/1` must produce a
router event. The normal `llm.step()` that follows performs the baseline
decode. Do not increment `commit_attempts` for baseline routes; add separate
`route_attempts`.

- [ ] **Step 5: Add aggregate route evidence**

Add per-prompt and summary fields:

```python
"route_attempts": len(route_events),
"route_counts": {
    name: sum(event["route"] == name for event in route_events)
    for name in (
        "baseline_short_draft",
        "baseline_finished",
        "baseline_output_budget",
        "baseline_incompatible",
        "native_multi_token",
    )
},
"fallback_reason_counts": {
    reason: sum(
        event.get("route_fallback_reason") == reason
        for event in route_events
    )
    for reason in sorted({
        event.get("route_fallback_reason")
        for event in route_events
        if event.get("route_fallback_reason")
    })
},
"router_events": route_events,
```

Include source proposal lookup/build time in end-to-end elapsed time. Do not
subtract it from the candidate result.

- [ ] **Step 6: Run focused and native regression tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_speculation_router.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_model_runner_spec_verify.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_oracle.py
python3 -m py_compile \
  tools/profile_ngram_commit.py \
  tools/test_speculation_router.py
```

Expected: all suites pass.

- [ ] **Step 7: Commit profiler routing**

```bash
git add \
  tools/profile_ngram_commit.py \
  tools/test_speculation_router.py \
  tools/test_ngram_speculative.py
git commit -m "feat: route short speculative drafts to baseline"
```

---

### Task 4: Reusable Source-Audit Module

**Files:**
- Create: `tools/source_audit.py`
- Modify: `tools/adaptive_ngram_gate.py`
- Modify: `tools/test_adaptive_ngram_gate.py`
- Create: `tools/test_source_audit.py`

**Interfaces:**
- Produces:
  `expand_owned_source_paths(repo_root: Path, owned_roots: tuple[str, ...], ignored_untracked_prefixes: tuple[str, ...] = ()) -> tuple[str, ...]`
- Produces:
  `hash_source_tree(source_root: Path, relative_paths: tuple[str, ...]) -> list[dict]`
- Produces:
  `source_tree_sha256(files: list[dict]) -> str`
- Produces:
  `build_source_evidence(repo_root: Path, out_dir: Path, *, owned_roots: tuple[str, ...], ignored_untracked_prefixes: tuple[str, ...] = ()) -> dict`
- Produces:
  `validate_source_snapshot(source_root: Path, evidence: dict, patch_path: Path, *, expected_owned_roots: tuple[str, ...]) -> dict`
- Produces:
  `reconstruct_source_snapshot(repo_root: Path, source_root: Path, evidence: dict, patch_path: Path, *, expected_owned_roots: tuple[str, ...]) -> None`
- Preserves adaptive-gate public names through imports or thin wrappers.

- [ ] **Step 1: Add failing generic source-audit tests**

Create `tools/test_source_audit.py` with a temporary Git repository containing
`tinyvllm/`, one router tool, and one unrelated untracked file. Assert:

```python
evidence = audit.build_source_evidence(
    root,
    root / "snapshot",
    owned_roots=(
        "tinyvllm",
        "tools/profile_ngram_commit.py",
        "tools/speculation_router_gate.py",
    ),
    ignored_untracked_prefixes=("experiments/speculation_router/",),
)
assert evidence["owned_roots"] == [
    "tinyvllm",
    "tools/profile_ngram_commit.py",
    "tools/speculation_router_gate.py",
]
assert evidence["tree_sha256"] == audit.source_tree_sha256(
    evidence["files"]
)
```

Also assert changed files, missing files, symlinks, patch tampering, tree-hash
tampering, and an untracked owned file are rejected.

- [ ] **Step 2: Run generic test and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_source_audit.py
```

Expected: import failure for `tools.source_audit`.

- [ ] **Step 3: Extract generic code without schema changes**

Move the generic implementations from `tools/adaptive_ngram_gate.py` to
`tools/source_audit.py`. Parameterize owned roots and ignored generated
prefixes. Keep JSON field names and canonical hashing byte-for-byte
compatible.

In `tools/adaptive_ngram_gate.py`, retain:

```python
from source_audit import (
    hash_source_tree,
    reconstruct_source_snapshot as _reconstruct_source_snapshot,
    source_tree_sha256,
    validate_source_snapshot as _validate_source_snapshot,
)
```

Add thin wrappers that pass the existing `OWNED_SOURCE_ROOTS` and adaptive
generated-artifact exemptions so existing callers and tests remain unchanged.

- [ ] **Step 4: Prove adaptive artifact compatibility**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_source_audit.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_adaptive_ngram_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/adaptive_ngram_gate.py verify \
  --out-dir experiments/adaptive_ngram/20260717-k1-sam-canonical
```

Expected:

- both test suites pass;
- canonical verifier still prints
  `ADAPTIVE_NGRAM_CANONICAL_AUDIT_OK NO_GO`.

- [ ] **Step 5: Commit the source-audit extraction**

```bash
git add \
  tools/source_audit.py \
  tools/test_source_audit.py \
  tools/adaptive_ngram_gate.py \
  tools/test_adaptive_ngram_gate.py
git commit -m "refactor: share source-auditable gate evidence"
```

---

### Task 5: Controlled Gate Schema and Classification

**Files:**
- Create: `tools/speculation_router_gate.py`
- Create: `tools/test_speculation_router_gate.py`

**Interfaces:**
- Produces:
  `CONTROLLED_POLICIES = ("baseline", "legacy_rematerialize", "always_native", "routed_native", "oracle")`
- Produces:
  `CONTROLLED_THRESHOLDS`
- Produces:
  `CONTROLLED_CASE_MATRIX`
- Produces:
  `build_controlled_manifest(*, source_evidence: dict, source_preflight: dict, model_path: str, model_identifier: str, host: str, python_bin: str, torch_version: str, cuda_version: str, flash_attn_version: str, gpu_name: str, bf16_supported: bool, run_tag: str) -> dict`
- Produces:
  `classify_controlled_gate(manifest: dict, capability: dict, case_rows: list[dict], event_rows: list[dict], router_rows: list[dict]) -> dict`
- Produces:
  `render_controlled_report(manifest: dict, capability: dict, case_rows: list[dict], event_rows: list[dict], router_rows: list[dict], summary: dict) -> str`
- Produces:
  `verify_artifacts(out_dir: Path) -> dict`
- Produces CLI:
  `python3 tools/speculation_router_gate.py snapshot-source --repo-root PATH --out-dir PATH`
- Produces CLI:
  `python3 tools/speculation_router_gate.py verify --out-dir PATH`

- [ ] **Step 1: Write synthetic complete controlled fixtures**

In `tools/test_speculation_router_gate.py`, add helpers that create:

- one `k1-route-fallback` row for `routed_native`;
- `K in {2,4,8,16}` × zero/one/partial/full acceptance rows;
- block-boundary, multi-block, EOS, budget, and 16-token continuation
  evidence;
- all five policies;
- one native/router event per required case;
- unique process port pairs;
- source tree identity on every row.

Set elapsed values so full/partial accepted `K=4/8/16` routed rows are `0.90`
of baseline, zero/one accepted rows are `1.00`, and all native events have
zero replay/copy/rematerialization.

- [ ] **Step 2: Write exact classification tests**

Add:

```python
def test_complete_controlled_evidence_is_ready():
    manifest, capability, rows, events, router = _controlled_fixture()
    summary = gate.classify_controlled_gate(
        manifest, capability, rows, events, router
    )
    assert summary["classification"] == "READY_FOR_REAL_DRAFTER_GATE"
    assert summary["exactness_pass"] is True
    assert summary["replay_elimination_pass"] is True
    assert summary["router_isolation_pass"] is True


def test_no_profitable_region_is_no_go():
    manifest, capability, rows, events, router = _controlled_fixture()
    for row in rows:
        if row["policy"] == "routed_native":
            row["elapsed_s"] = row["baseline_elapsed_s"]
    assert gate.classify_controlled_gate(
        manifest, capability, rows, events, router
    )["classification"] == "NO_GO"


def test_short_route_mutation_is_no_go():
    manifest, capability, rows, events, router = _controlled_fixture()
    short = next(row for row in router if row["draft_len"] == 1)
    short["speculative_reservation_attempted"] = True
    assert gate.classify_controlled_gate(
        manifest, capability, rows, events, router
    )["classification"] == "NO_GO"
```

Also test semantic mismatch, replay, copy, rematerialization, lifecycle
regression above `1.05`, missing row, duplicate row, failed process, duplicate
ports, unavailable capability, non-finite timing, source mismatch, report
drift, and hash tampering.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_speculation_router_gate.py
```

Expected: import failure for `tools.speculation_router_gate`.

- [ ] **Step 4: Implement constants and deterministic case matrix**

Use:

```python
CONTROLLED_THRESHOLDS = {
    "profitable_region_max_elapsed_ratio": 0.95,
    "max_required_lifecycle_elapsed_ratio": 1.05,
    "min_continuation_steps": 16,
}
```

Build deterministic cases with stable IDs, prompt SHA-256, target-derived
draft labels, history lengths, block cases, expected accepted counts, and
continuation steps. Include `k1-route-fallback` only as a router-isolation
case; performance-region aggregation uses `K>=2`.

- [ ] **Step 5: Implement fail-closed controlled classification**

Classification order:

1. source/manifest/row/process/port/event completeness → `INCOMPLETE`;
2. capability completeness → `INCOMPLETE`;
3. exactness and continuation → `NO_GO`;
4. native replay/copy/rematerialization → `NO_GO`;
5. short-route mutation or target work → `NO_GO`;
6. finite performance evidence → `INCOMPLETE`;
7. required lifecycle ratio above `1.05` → `NO_GO`;
8. no accepted `K>=2` region below `0.95` → `NO_GO`;
9. otherwise `READY_FOR_REAL_DRAFTER_GATE`.

Do not remove zero- or one-acceptance rows from completeness or lifecycle
aggregation.

- [ ] **Step 6: Implement canonical report and artifact verification**

Required files for the controlled stage:

```python
CONTROLLED_REQUIRED_ARTIFACTS = (
    "source_evidence.json",
    "source.patch",
    "source_snapshot.tar.gz",
    "source_preflight.json",
    "manifest.json",
    "capability.json",
    "case_rows.json",
    "event_rows.json",
    "router_rows.json",
    "summary.json",
    "report.md",
    "artifact_hashes.json",
    "remote_exitcode",
    "runner.log",
)
```

`verify_artifacts()` must reconstruct source, verify all hashes, recompute the
summary from raw rows, render the report again, and reject any disagreement.

Add the `snapshot-source` subcommand as a thin CLI over
`source_audit.build_source_evidence()` using the router gate's fixed owned
source roots. Add `verify` as a thin CLI over `verify_artifacts()`. The
subcommands print sorted JSON and exit nonzero on validation failure.

- [ ] **Step 7: Run synthetic gate tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_speculation_router_gate.py
python3 -m py_compile \
  tools/speculation_router_gate.py \
  tools/test_speculation_router_gate.py
```

Expected: all synthetic tests pass.

- [ ] **Step 8: Commit controlled gate core**

```bash
git add \
  tools/speculation_router_gate.py \
  tools/test_speculation_router_gate.py
git commit -m "feat: add controlled speculation router gate"
```

---

### Task 6: Routed Runtime Policies and Controlled Process Driver

**Files:**
- Modify: `tools/native_verifier_oracle.py`
- Modify: `tools/test_native_verifier_oracle.py`
- Modify: `tools/speculation_router_gate.py`
- Modify: `tools/test_speculation_router_gate.py`

**Interfaces:**
- Extends oracle policy choices with `always_native` and `routed_native`.
- Produces case payload fields:
  `route`, `route_fallback_reason`, `router_event`,
  `draft_construction="controlled_target_derived"`.
- Produces:
  `run_controlled_gate(*, out_dir: Path, python_bin: str, model_path: str, source_evidence_path: Path, source_patch_path: Path, source_preflight_path: Path, host: str, run_tag: str, resume: bool = False, case_limit: int = 0) -> dict`

- [ ] **Step 1: Add failing oracle policy tests**

Add synthetic tests that monkeypatch the profile wrapper:

```python
def test_routed_native_payload_records_short_fallback():
    payload = {
        "case_id": "k1-route-fallback",
        "policy": "routed_native",
        "status": "PASS",
        "event": {
            "route": "baseline_short_draft",
            "accepted_count": 0,
            "target_forward_count": 0,
        },
    }
    assert payload["event"] == {
        "route": "baseline_short_draft",
        "accepted_count": 0,
        "target_forward_count": 0,
    }


def test_routed_native_multi_token_retains_oracle_evidence():
    native = _comparison_payload(
        policy="routed_native",
        route="native_multi_token",
        target_tokens=[10, 20, 30, 40],
        accepted_tokens=[10, 20, 30, 40],
    )
    serialized = _comparison_payload(
        policy="oracle",
        route=None,
        target_tokens=[10, 20, 30, 40],
        accepted_tokens=[10, 20, 30, 40],
    )
    comparison = oracle.compare_native_and_oracle(
        native,
        serialized,
    )
    assert comparison["status"] == "PASS"
```

Define `_comparison_payload()` in the same test file by reusing the existing
complete native/oracle fixture fields and varying only `policy`, `route`,
`target_tokens`, and `accepted_tokens`. Test that controlled target-derived
draft materialization records the exact expected accepted count and cannot be
selected in real-source mode.

- [ ] **Step 2: Run oracle tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_oracle.py
```

Expected: failure for unknown routed policy or missing route evidence.

- [ ] **Step 3: Add routed policies without changing existing policies**

In `_run_tinyvllm_case()`:

- `always_native` calls `verify_and_commit_block()` with
  `verifier_mode="native"`;
- `routed_native` calls
  `route_and_verify_draft()` with
  `allow_incompatible_fallback=False`;
- existing `native`, `legacy_rematerialize`, `baseline`, `probe`, and
  serialized `oracle` behavior remains unchanged.

For a baseline route, do not claim accepted draft tokens. Allow the normal
decode continuation to produce the next target token and record the route as
zero speculative target forwards.

- [ ] **Step 4: Add isolated process execution and resume**

In `tools/speculation_router_gate.py`, follow the native gate process pattern:

- one probe process per case;
- one isolated process per policy/case;
- two dynamic ports per process;
- retry only failed or incomplete rows;
- successful rows are immutable on `--resume`;
- replacing a row also replaces stale event/router rows;
- process stdout/stderr live under run-local `logs/`;
- raw payloads live under run-local `raw/`.

Normalize large tensor evidence out of top-level rows where possible. Keep
oracle numeric arrays only in raw payloads and store hashes/max errors in
`case_rows.json` to avoid another multi-gigabyte aggregate file.

- [ ] **Step 5: Add process-driver tests**

Monkeypatch the subprocess runner and assert:

- all expected keys run;
- each port pair is unique;
- `--case-limit` selects a deterministic prefix for smoke only;
- a failed row is retried by `--resume`;
- a successful row is not rerun;
- stale event/router rows are removed when a failed row is replaced;
- controlled mode rejects a non-target-derived construction label.

- [ ] **Step 6: Run focused and regression suites**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_oracle.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_speculation_router_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_gate.py
python3 -m py_compile \
  tools/native_verifier_oracle.py \
  tools/speculation_router_gate.py
```

Expected: all suites pass.

- [ ] **Step 7: Commit controlled runtime execution**

```bash
git add \
  tools/native_verifier_oracle.py \
  tools/test_native_verifier_oracle.py \
  tools/speculation_router_gate.py \
  tools/test_speculation_router_gate.py
git commit -m "feat: execute routed verifier envelope cases"
```

---

### Task 7: Generic Real-Source Gate

**Files:**
- Modify: `tools/speculation_router_gate.py`
- Modify: `tools/test_speculation_router_gate.py`
- Modify: `tools/profile_ngram_commit.py`
- Modify: `tools/test_speculation_router.py`

**Interfaces:**
- Produces:
  `REAL_POLICIES = ("baseline", "source_always_native", "source_routed_native")`
- Produces:
  `REAL_THRESHOLDS`
- Produces:
  `validate_draft_source_manifest(draft_source: dict) -> None`
- Produces:
  `classify_real_source_gate(manifest: dict, draft_source: dict, prompt_bank: dict, case_rows: list[dict], event_rows: list[dict], router_rows: list[dict]) -> dict`
- Produces:
  `run_real_source_gate(*, out_dir: Path, python_bin: str, model_path: str, source_evidence_path: Path, source_patch_path: Path, source_preflight_path: Path, draft_source_path: Path, prompt_bank_path: Path, host: str, run_tag: str, repetitions: int, warmup_repetitions: int, resume: bool = False, prompt_limit: int = 0) -> dict`
- Produces CLI:
  `python3 tools/speculation_router_gate.py validate-real-input --draft-source PATH --prompt-bank PATH`

- [ ] **Step 1: Add exact real-source manifest tests**

Use this minimum schema:

```python
{
    "schema_version": 1,
    "source_name": "fixture-learned-drafter",
    "source_type": "learned_speculative_head",
    "implementation_paths": ["tools/profile_ngram_commit.py"],
    "source_tree_sha256": "a" * 64,
    "checkpoint_identifier": "fixture/checkpoint",
    "checkpoint_config_sha256": "b" * 64,
    "tokenizer_identifier": "Qwen3-0.6B",
    "vocab_size": 151936,
    "hyperparameters": {"max_draft_tokens": 8},
    "consumes_target_hidden_states": True,
    "requires_additional_model_forward": True,
    "target_derived": False,
    "debug_stub": False,
    "prompt_bank_sha256": "c" * 64,
}
```

Reject `target_derived=true`, `debug_stub=true`, missing checkpoint identity
for learned sources, mismatched prompt-bank hash, unknown source type, and
mutable post-run hyperparameters.

- [ ] **Step 2: Add synthetic real-source classification tests**

Build rows for natural, code, repetitive, transition-heavy, low-match, EOS,
short-context, and long-context buckets. Add tests for:

- complete `GO`;
- aggregate elapsed gain below `5%` → `NO_GO`;
- tokens/s gain below `5%` → `NO_GO`;
- natural or transition-heavy ratio above `1.00` → `NO_GO`;
- one prompt regression above `10%` → `NO_GO`;
- routed slower than always-native → `NO_GO`;
- missing both native and fallback route exercise → `NO_GO`;
- non-positive target-forward reduction → `NO_GO`;
- output mismatch → `NO_GO`;
- missing source/checkpoint/prompt identity → `INCOMPLETE`;
- failed process or non-finite timing → `INCOMPLETE`.

- [ ] **Step 3: Implement real-source validation and classification**

Use:

```python
REAL_THRESHOLDS = {
    "min_elapsed_improvement_fraction": 0.05,
    "min_tokens_per_s_improvement_fraction": 0.05,
    "max_natural_elapsed_ratio": 1.00,
    "max_transition_elapsed_ratio": 1.00,
    "max_individual_prompt_elapsed_ratio": 1.10,
}
```

Classification order remains structural `INCOMPLETE`, semantic/replay
`NO_GO`, then complete performance `NO_GO`, then `GO`.

- [ ] **Step 4: Add real-source profiler evidence**

For candidate-only runs, record:

- proposal construction start/end and elapsed time;
- exact proposed tokens;
- source metadata hash;
- route decision;
- verifier target-forward count;
- accepted/rejected counts;
- fallback reason;
- end-to-end elapsed including proposal construction.

Add an explicit `--gate-stage controlled|real-source` argument. Reject
`controlled_target_derived` when `gate-stage=real-source`.

- [ ] **Step 5: Implement generic real-source process orchestration**

`run_real_source_gate()` reads immutable `draft_source.json` and
`prompt_bank.json`, launches the three policies with identical prompt,
output-budget, seed, dtype, warmup, and repetition settings, and writes the
same core artifacts plus:

- `draft_source.json`;
- `prompt_bank.json`;
- `prompt_bank.sha256`.

The implementation supports an already-implemented named source. It does not
download or train a source.

Add `validate-real-input` to load both files, recompute the canonical
prompt-bank SHA-256, call `validate_draft_source_manifest()`, print:

```json
{
  "status": "PASS",
  "source_name": "<source_name>",
  "prompt_bank_sha256": "<64-hex>"
}
```

and exit nonzero for any schema or identity failure.

- [ ] **Step 6: Run tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_speculation_router.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_speculation_router_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
python3 -m py_compile \
  tools/profile_ngram_commit.py \
  tools/speculation_router_gate.py
```

Expected: all suites pass.

- [ ] **Step 7: Commit real-source gate support**

```bash
git add \
  tools/profile_ngram_commit.py \
  tools/test_speculation_router.py \
  tools/speculation_router_gate.py \
  tools/test_speculation_router_gate.py
git commit -m "feat: add source-attributed speculation performance gate"
```

---

### Task 8: Source-Auditable Remote Runner

**Files:**
- Create: `tools/run_speculation_router_gate_remote.sh`
- Create: `tools/test_run_speculation_router_gate_remote.py`
- Modify: `tools/speculation_router_gate.py`
- Modify: `tools/test_speculation_router_gate.py`

**Interfaces:**
- CLI:
  `tools/run_speculation_router_gate_remote.sh preflight`
- CLI:
  `tools/run_speculation_router_gate_remote.sh controlled-smoke`
- CLI:
  `tools/run_speculation_router_gate_remote.sh controlled`
- CLI:
  `tools/run_speculation_router_gate_remote.sh real-smoke DRAFT_SOURCE_JSON PROMPT_BANK_JSON`
- CLI:
  `tools/run_speculation_router_gate_remote.sh real DRAFT_SOURCE_JSON PROMPT_BANK_JSON`
- CLI:
  `python3 tools/speculation_router_gate.py verify --out-dir PATH`

- [ ] **Step 1: Write failing shell-contract tests**

Create `tools/test_run_speculation_router_gate_remote.py` and assert the
script contains:

```python
assert "sitian@10.232.195.203" in runner
assert "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python" in runner
assert "/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B" in runner
assert "snapshot-source" in runner
assert "source_evidence.json" in runner
assert "source.patch" in runner
assert "source_snapshot.tar.gz" in runner
assert "source_preflight.json" in runner
assert "TINYVLLM_DIST_PORT" in gate_source
assert "MASTER_PORT" in gate_source
assert "nohup" in runner
assert "remote_exitcode" in runner
assert "runner.log" in runner
assert "verify --out-dir" in runner
```

Also assert the owned source list includes router, profiler, source audit,
gate, oracle, tests, and runner.

- [ ] **Step 2: Run shell tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python3 tools/test_run_speculation_router_gate_remote.py
```

Expected: missing runner file failure.

- [ ] **Step 3: Implement immutable local staging**

The runner:

1. creates a run-local staging directory;
2. calls:

```bash
python3 tools/speculation_router_gate.py snapshot-source \
  --repo-root "${REPO_ROOT}" \
  --out-dir "${STAGING_DIR}"
```
3. records base commit, patch hash, and tree hash;
4. uploads only the staged source tree and evidence;
5. never rereads mutable owned worktree files for the payload;
6. rejects unrelated tracked/staged/unstaged changes outside owned roots;
7. ignores only generated artifacts under
   `experiments/speculation_router/` and the retained adaptive artifact
   directories.

- [ ] **Step 4: Implement remote preflight**

Remote preflight must:

- verify model config identifies Qwen3-0.6B;
- verify staged source hashes before importing TinyLLMForge;
- record Python, torch, CUDA, FlashAttention, GPU, host, model path,
  `tinyvllm.__file__`, source tree SHA-256, and patch SHA-256;
- run `py_compile`;
- run dependency-light router, gate, oracle, native verifier, and attention
  tests;
- run the FlashAttention capability matrix;
- use a run-local `TMPDIR`.

- [ ] **Step 5: Implement detached model execution and artifact retrieval**

For controlled/real canonical runs, launch:

```bash
nohup env \
  TMPDIR="${REMOTE_DIR}/tmp" \
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="${REMOTE_DIR}/source" \
  "${REMOTE_PYTHON}" \
  tools/speculation_router_gate.py run-controlled \
    --out-dir "${REMOTE_DIR}/artifacts" \
    --python-bin "${REMOTE_PYTHON}" \
    --model-path "${MODEL_PATH}" \
    --source-evidence "${REMOTE_DIR}/source_evidence.json" \
    --source-patch "${REMOTE_DIR}/source.patch" \
    --source-preflight "${REMOTE_DIR}/source_preflight.json" \
    --host "${REMOTE_HOST}" \
    --run-tag "${RUN_TAG}" \
    ${CASE_LIMIT:+--case-limit "${CASE_LIMIT}"} \
  >"${REMOTE_DIR}/runner.log" 2>&1 &
```

The detached wrapper writes `remote_exitcode` atomically. The local script
polls only that run-local file. It does not treat SSH disconnect as model
failure.

Download canonical files individually, verify sizes/hashes, then run the
local verifier. Do not create a huge intermediate tarball containing raw
tensor arrays.

- [ ] **Step 6: Add runner and verifier negative tests**

Test:

- local source changes after staging do not affect uploaded bytes;
- remote preflight tree mismatch stops before model process;
- missing exit code remains `INCOMPLETE`;
- nonzero remote exit preserves logs/artifacts;
- truncated JSON is rejected;
- missing canonical file is rejected;
- source snapshot or patch tampering is rejected;
- controlled target-derived drafts cannot enter real mode.

- [ ] **Step 7: Run local runner validation**

Run:

```bash
bash -n tools/run_speculation_router_gate_remote.sh
PYTHONDONTWRITEBYTECODE=1 \
  python3 tools/test_run_speculation_router_gate_remote.py
PYTHONDONTWRITEBYTECODE=1 \
  python3 tools/test_speculation_router_gate.py
PYTHONDONTWRITEBYTECODE=1 \
  python3 tools/speculation_router_gate.py --help
git diff --check
```

Expected: all commands exit `0`.

- [ ] **Step 8: Commit remote orchestration**

```bash
git add \
  tools/run_speculation_router_gate_remote.sh \
  tools/test_run_speculation_router_gate_remote.py \
  tools/speculation_router_gate.py \
  tools/test_speculation_router_gate.py
git commit -m "feat: add auditable speculation router remote gate"
```

---

### Task 9: Full Local Regression Gate

**Files:**
- Verify all implementation files from Tasks 2–8.

**Interfaces:**
- Produces: fresh local evidence before any remote model run.

- [ ] **Step 1: Run all focused dependency-light tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_speculation_router.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_source_audit.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_speculation_router_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_run_speculation_router_gate_remote.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_contract.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_model_runner_spec_verify.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_attention.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_oracle.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_adaptive_ngram_gate.py
```

Expected: every suite prints its pass sentinel and exits `0`.

- [ ] **Step 2: Run compilation, shell, artifact, and diff checks**

```bash
python3 -m py_compile \
  tinyvllm/speculative/router.py \
  tools/profile_ngram_commit.py \
  tools/source_audit.py \
  tools/speculation_router_gate.py \
  tools/native_verifier_oracle.py
bash -n tools/run_speculation_router_gate_remote.sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/adaptive_ngram_gate.py verify \
  --out-dir experiments/adaptive_ngram/20260717-k1-sam-canonical
git diff --check
git status --short
```

Expected:

- compilation and shell syntax pass;
- adaptive canonical remains `NO_GO` with the same evidence;
- no failed K1 implementation diff has reappeared;
- only intended source changes and retained artifacts are present.

- [ ] **Step 3: Commit any test-only corrections**

If Step 1 or 2 required a focused correction, stage only its owned files and
commit:

```bash
git commit -m "test: harden speculation router regression gate"
```

If no corrections were required, do not create an empty commit.

---

### Task 10: Remote Controlled Smoke and Canonical Envelope

**Files:**
- Generate:
  `experiments/speculation_router/<controlled-smoke-run-tag>/`
- Generate:
  `experiments/speculation_router/<controlled-canonical-run-tag>/`

**Interfaces:**
- Produces the first real classification:
  `READY_FOR_REAL_DRAFTER_GATE`, `NO_GO`, or `INCOMPLETE`.

- [ ] **Step 1: Verify remote access without changing remote state**

Run:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 \
  -S /tmp/ssh-sitian-10.232.195.203 \
  sitian@10.232.195.203 \
  'hostname; nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader'
```

Expected: host and GPU inventory. If authentication fails, record the exact
error and stop remote work without changing classification.

- [ ] **Step 2: Run remote preflight**

Run:

```bash
RUN_TAG="qwen3-06b-router-preflight-$(date +%Y%m%d-%H%M%S)" \
  tools/run_speculation_router_gate_remote.sh preflight
```

Expected:

- source snapshot verifies locally and remotely;
- remote dependency-light tests pass;
- FP16 and supported BF16 capability rows pass;
- local preflight artifact verifies.

- [ ] **Step 3: Run controlled smoke**

Run:

```bash
RUN_TAG="qwen3-06b-router-controlled-smoke-$(date +%Y%m%d-%H%M%S)" \
CASE_LIMIT=6 \
  tools/run_speculation_router_gate_remote.sh controlled-smoke
```

The deterministic smoke prefix must include:

- one `K=1` baseline fallback;
- one zero-accept native case;
- one partial-accept native case;
- one full-accept native case;
- one block boundary;
- one continuation comparison.

Expected: local artifact verification succeeds. A reduced smoke may be
provisionally `NO_GO`, but it must not be structurally `INCOMPLETE`.

- [ ] **Step 4: Audit smoke coverage before canonical**

Run:

```bash
python3 tools/speculation_router_gate.py verify \
  --out-dir "experiments/speculation_router/${RUN_TAG}"
jq '.classification,.reasons,.observed_case_rows,.observed_router_rows' \
  "experiments/speculation_router/${RUN_TAG}/summary.json"
```

Expected: source, exactness, router isolation, replay elimination, ports, and
required smoke branches verify.

- [ ] **Step 5: Run full controlled canonical**

Run:

```bash
RUN_TAG="qwen3-06b-router-controlled-canonical-$(date +%Y%m%d-%H%M%S)" \
  tools/run_speculation_router_gate_remote.sh controlled
```

Poll the run-local `remote_exitcode` until completion. If transient GPU
allocation fails, use `--resume` against the immutable same-source manifest;
do not change thresholds or case order.

- [ ] **Step 6: Independently verify canonical evidence**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python3 tools/speculation_router_gate.py verify \
    --out-dir "experiments/speculation_router/${RUN_TAG}"
python3 - <<'PY' "experiments/speculation_router/${RUN_TAG}"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
summary = json.loads((root / "summary.json").read_text())
pairs = [
    (row["tinyvllm_dist_port"], row["master_port"])
    for row in manifest["process_port_pairs"]
]
assert len(pairs) == len(set(pairs))
assert summary["classification"] in {
    "READY_FOR_REAL_DRAFTER_GATE",
    "NO_GO",
}
print("SPECULATION_ROUTER_CANONICAL_AUDIT_OK",
      summary["classification"])
PY
```

Expected: never `INCOMPLETE`.

- [ ] **Step 7: Follow the measured branch**

If `READY_FOR_REAL_DRAFTER_GATE`:

- preserve the artifact;
- do not claim product `GO`;
- inspect locally/officially available compatible learned drafter
  checkpoints;
- start a new brainstorming design for one attributable EAGLE-style or
  smaller-model source.

If `NO_GO`:

- preserve the negative artifact;
- stop native-verifier/router micro-optimization;
- shift the next optimization search toward a different engine bottleneck
  such as production batching, kernel/graph overhead, or quantization, each
  behind its own written design and gate.

If `INCOMPLETE`:

- repair evidence/infrastructure only;
- retain thresholds and immutable source;
- resume the same manifest.

---

### Task 11: Real-Source Smoke When an Eligible Source Exists

**Files:**
- Create through a separately approved source design:
  `draft_source.json`
- Create through that design:
  `prompt_bank.json`
- Generate:
  `experiments/speculation_router/<real-source-run-tag>/`

**Interfaces:**
- Consumes: an implemented, source-attributed, non-target-derived source.
- Produces: `GO`, `NO_GO`, or `INCOMPLETE`.

This task is conditional. Do not execute it with a debug stub or a renamed
existing negative-control source.

- [ ] **Step 1: Validate source and prompt manifests locally**

Run:

```bash
python3 tools/speculation_router_gate.py validate-real-input \
  --draft-source draft_source.json \
  --prompt-bank prompt_bank.json
```

Expected: source is non-target-derived, non-debug, checkpoint/config/tokenizer
identity is complete, and the prompt-bank hash matches.

- [ ] **Step 2: Run real-source smoke**

```bash
RUN_TAG="qwen3-06b-router-real-smoke-$(date +%Y%m%d-%H%M%S)" \
  tools/run_speculation_router_gate_remote.sh \
    real-smoke draft_source.json prompt_bank.json
```

Expected: baseline, always-native, and routed policies all complete with
exact output and both native/fallback route exercise.

- [ ] **Step 3: Run canonical only after smoke verification**

```bash
RUN_TAG="qwen3-06b-router-real-canonical-$(date +%Y%m%d-%H%M%S)" \
  tools/run_speculation_router_gate_remote.sh \
    real draft_source.json prompt_bank.json
```

- [ ] **Step 4: Verify and preserve the measured decision**

```bash
python3 tools/speculation_router_gate.py verify \
  --out-dir "experiments/speculation_router/${RUN_TAG}"
```

Interpret only the recorded `GO`, `NO_GO`, or `INCOMPLETE`; do not retune the
same evaluation prompts after seeing results.

---

### Task 12: Documentation, Handoff, and Completion Audit

**Files:**
- Modify: `README.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Verify:
  `docs/superpowers/specs/2026-07-17-speculation-profitability-router-gate-design.md`
- Verify:
  `docs/superpowers/plans/2026-07-17-speculation-profitability-router-gate.md`

**Interfaces:**
- Produces: durable reproduction commands, exact measured classification,
  evidence paths, claim boundaries, and next direction.

- [ ] **Step 1: Update README from real artifacts**

Document:

- fixed router semantics;
- controlled smoke/canonical commands;
- local verifier command;
- exact run tag, source commit/tree/patch hashes;
- case/row/event/port counts;
- exact classification and reasons;
- per-region routed/baseline ratios;
- claim boundaries;
- conditional real-source command.

Do not write `GO` or speedup numbers before the corresponding artifact exists.

- [ ] **Step 2: Update the handoff**

Record in `AGENT_HANDOFF_STATE.md`:

- failed K1 retirement and preserved artifact;
- implementation commits;
- local tests and exact pass sentinels;
- remote host/Python/model/GPU;
- smoke and canonical paths;
- source identity;
- retry/resume history;
- final controlled classification;
- what it proves and does not prove;
- exact next branch from Task 10 Step 7.

- [ ] **Step 3: Build the prompt-to-artifact completion checklist**

Add a checklist mapping every design requirement:

```text
[ ] K<=1 baseline isolation -> router_rows.json + wrapper tests
[ ] K>=2 native dispatch -> event_rows.json + oracle comparison
[ ] zero replay/copy/rematerialization -> event rows + classifier
[ ] K 2/4/8/16 acceptance matrix -> manifest + case rows
[ ] EOS/budget/block/continuation -> case rows + oracle fields
[ ] controlled threshold -> recomputed summary
[ ] source identity -> source evidence/preflight/reconstruction
[ ] dynamic unique ports -> manifest independent audit
[ ] real-source-only GO boundary -> classifier tests
[ ] remote result -> summary/report/local verifier
[ ] limitations and next direction -> README/handoff
```

Every item must point to a concrete file, JSON key, test, or command output.
Unchecked items mean the objective is not complete.

- [ ] **Step 4: Run final fresh verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_speculation_router.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_source_audit.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_speculation_router_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_run_speculation_router_gate_remote.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_oracle.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_adaptive_ngram_gate.py
python3 -m py_compile \
  tinyvllm/speculative/router.py \
  tools/profile_ngram_commit.py \
  tools/source_audit.py \
  tools/speculation_router_gate.py \
  tools/native_verifier_oracle.py
bash -n tools/run_speculation_router_gate_remote.sh
python3 tools/speculation_router_gate.py verify \
  --out-dir "experiments/speculation_router/${CONTROLLED_CANONICAL_RUN_TAG}"
git diff --check
```

Expected: all local tests and artifact verification pass.

- [ ] **Step 5: Commit measured documentation**

```bash
git add \
  README.md \
  AGENT_HANDOFF_STATE.md
git commit -m "docs: record speculation router gate result"
```

- [ ] **Step 6: Final scope audit**

Run:

```bash
git status --short
git log --oneline --decorate -12
git diff HEAD^ --check
```

Confirm:

- no failed K1 implementation is committed;
- adaptive artifacts remain preserved;
- controlled/real artifacts remain preserved;
- no production claim exceeds the measured gate;
- all explicit design requirements are mapped to evidence;
- unresolved real-source work is stated as unresolved rather than complete.
