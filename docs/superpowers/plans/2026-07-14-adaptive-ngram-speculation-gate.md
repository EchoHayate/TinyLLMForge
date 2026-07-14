# Adaptive N-Gram Speculation Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible single-sequence Qwen3-0.6B gate that compares an adaptive `K in {1,2,4}` n-gram draft cap against isolated normal greedy and fixed `K=1/2/4` policies without changing target verification or commit correctness.

**Architecture:** Add a pure CPU-only adaptive state machine to `tinyvllm/speculative/ngram.py`, then let `tools/profile_ngram_commit.py` use that state only to choose the next proposal cap. A separate dependency-light gate module owns the committed prompt bank, subprocess matrix, raw artifact schema, trajectory replay, median aggregation, fixed decision thresholds, and report generation; a shell runner uploads the minimum isolated source tree to `sitian@10.232.195.203`, discovers the real Qwen3-0.6B path, assigns fresh ports per model process, and downloads canonical artifacts.

**Tech Stack:** Python 3 dataclasses and standard library (`argparse`, `hashlib`, `json`, `math`, `pathlib`, `random`, `statistics`, `subprocess`, `tempfile`), existing TinyLLMForge profiler/engine, Bash, SSH/SCP, Qwen3-0.6B on the existing remote CUDA host.

## Global Constraints

- The written design is `docs/superpowers/specs/2026-07-14-adaptive-ngram-speculation-design.md`; its correctness and decision thresholds are normative.
- The first version is strictly greedy, single-sequence, and profiler-owned; do not modify `Sequence`, the scheduler, `LLM.generate()`, target logits, accepted-prefix logic, KV reservation/commit, EOS handling, or normal decode fallback.
- Adaptive levels are exactly `(1, 2, 4)`, initial `K=2`, initial EMA `0.5`, EMA weight `0.5`, promotion EMA threshold `0.75`, demotion threshold `0.5`, and promotion streak length `2`.
- A no-match position does not update adaptive state.
- Fixed `K=1/2/4`, adaptive, and normal greedy run in separate processes with exactly one active sequence (`--max-num-seqs 1` and one literal `--prompt`).
- Canonical coverage is four committed prompt classes, five policies, and seven repetitions, for `4 * 5 * 7 = 140` unique raw rows.
- Candidate ordering is deterministically shuffled within each repetition, and every model process receives distinct dynamic `TINYVLLM_DIST_PORT` and `MASTER_PORT` values.
- Remote execution uses `sitian@10.232.195.203` and `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`; discover and record the actual Qwen3-0.6B path instead of guessing its spelling.
- Upload into a new isolated remote directory; do not modify or trust a pre-existing remote checkout.
- Missing rows, duplicate keys, failed processes, invalid/non-finite timings, model discovery failure, or port collision produce `INCOMPLETE`, never `NO_GO`.
- Do not relax the committed `+5%`, `+2%` / near-best-plus-waste, `20%`, `15%`, and per-prompt `-5%` thresholds after observing results.
- Canonical artifacts live under `experiments/adaptive_ngram/<model-and-date>/` and include `manifest.json`, `raw_rows.json`, `event_rows.json`, `summary.json`, and `report.md`.
- Important commands, limitations, result interpretation, and next direction must also be written to `README.md` and `AGENT_HANDOFF_STATE.md`.

---

### Task 1: Pure Adaptive Draft State Machine

**Files:**
- Modify: `tinyvllm/speculative/ngram.py`
- Modify: `tools/test_ngram_speculative.py`

**Interfaces:**
- Produces: `AdaptiveDraftState(levels: tuple[int, ...] = (1, 2, 4), level_index: int = 1, acceptance_ema: float = 0.5, full_accept_streak: int = 0, proposal_events: int = 0)`
- Produces: `AdaptiveDraftState.selected_k -> int`
- Produces: `update_adaptive_draft_state(state: AdaptiveDraftState, proposed: int, accepted: int) -> dict[str, int | float | str | bool | list[int]]`
- The transition record contains enough pre/post state to replay every event without model state.

- [ ] **Step 1: Write failing transition and validation tests**

Add direct aliases near the existing n-gram aliases in `tools/test_ngram_speculative.py`:

```python
AdaptiveDraftState = ngram.AdaptiveDraftState
update_adaptive_draft_state = ngram.update_adaptive_draft_state
```

Add these tests:

```python
def test_adaptive_draft_state_starts_at_k2():
    state = AdaptiveDraftState()
    assert state.levels == (1, 2, 4)
    assert state.selected_k == 2
    assert state.acceptance_ema == 0.5
    assert state.full_accept_streak == 0
    assert state.proposal_events == 0


def test_adaptive_two_strong_full_accepts_promote_and_saturate():
    state = AdaptiveDraftState()
    first = update_adaptive_draft_state(state, proposed=2, accepted=2)
    second = update_adaptive_draft_state(state, proposed=2, accepted=2)
    third = update_adaptive_draft_state(state, proposed=4, accepted=4)
    fourth = update_adaptive_draft_state(state, proposed=4, accepted=4)

    assert first["selected_k_before"] == 2
    assert first["selected_k_after"] == 2
    assert first["transition_reason"] == "full_accept_streak"
    assert second["selected_k_after"] == 4
    assert second["transition_reason"] == "promote"
    assert fourth["selected_k_after"] == 4
    assert state.selected_k == 4


def test_adaptive_zero_accept_jumps_from_k4_to_k1():
    state = AdaptiveDraftState(level_index=2, acceptance_ema=0.9, full_accept_streak=1)
    event = update_adaptive_draft_state(state, proposed=4, accepted=0)
    assert event["selected_k_before"] == 4
    assert event["selected_k_after"] == 1
    assert event["transition_reason"] == "zero_accept"
    assert state.full_accept_streak == 0


def test_adaptive_weak_partial_accept_moves_down_one_level():
    state = AdaptiveDraftState(level_index=2, acceptance_ema=0.8)
    event = update_adaptive_draft_state(state, proposed=4, accepted=1)
    assert event["selected_k_after"] == 2
    assert event["transition_reason"] == "weak_acceptance"


def test_adaptive_weak_ema_demotes_k2_to_k1():
    state = AdaptiveDraftState(level_index=1, acceptance_ema=0.2)
    event = update_adaptive_draft_state(state, proposed=2, accepted=1)
    assert event["event_acceptance"] == 0.5
    assert event["acceptance_ema_after"] == 0.35
    assert event["selected_k_after"] == 1


def test_adaptive_partial_accept_resets_full_accept_streak():
    state = AdaptiveDraftState(full_accept_streak=1, acceptance_ema=0.9)
    event = update_adaptive_draft_state(state, proposed=2, accepted=1)
    assert event["selected_k_after"] == 2
    assert event["full_accept_streak_after"] == 0
    assert state.full_accept_streak == 0


def test_adaptive_rejects_invalid_counts_and_state():
    for proposed, accepted in ((0, 0), (-1, 0), (2, -1), (2, 3)):
        try:
            update_adaptive_draft_state(AdaptiveDraftState(), proposed, accepted)
        except ValueError:
            pass
        else:
            raise AssertionError((proposed, accepted))
    try:
        AdaptiveDraftState(levels=(1, 3, 4))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid adaptive levels accepted")


def test_adaptive_transition_record_is_json_friendly_and_replayable():
    import json
    state = AdaptiveDraftState()
    event = update_adaptive_draft_state(state, proposed=1, accepted=1)
    assert json.loads(json.dumps(event)) == event
    assert event == {
        "levels": [1, 2, 4],
        "proposal_event": 1,
        "proposed_tokens": 1,
        "accepted_tokens": 1,
        "event_acceptance": 1.0,
        "acceptance_ema_before": 0.5,
        "acceptance_ema_after": 0.75,
        "full_accept_streak_before": 0,
        "full_accept_streak_after": 1,
        "selected_k_before": 2,
        "selected_k_after": 2,
        "transition_reason": "full_accept_streak",
        "promoted": False,
        "demoted": False,
    }
```

- [ ] **Step 2: Run the focused tests and confirm the missing interface fails**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
```

Expected: failure while importing `AdaptiveDraftState` or `update_adaptive_draft_state`.

- [ ] **Step 3: Implement the minimal deterministic state machine**

Add after `NGramDraft` in `tinyvllm/speculative/ngram.py`:

```python
@dataclass
class AdaptiveDraftState:
    levels: tuple[int, ...] = (1, 2, 4)
    level_index: int = 1
    acceptance_ema: float = 0.5
    full_accept_streak: int = 0
    proposal_events: int = 0

    def __post_init__(self):
        if self.levels != (1, 2, 4):
            raise ValueError("adaptive levels must be exactly (1, 2, 4)")
        if self.level_index < 0 or self.level_index >= len(self.levels):
            raise ValueError("level_index is outside adaptive levels")
        if not 0.0 <= self.acceptance_ema <= 1.0:
            raise ValueError("acceptance_ema must be within [0, 1]")
        if self.full_accept_streak < 0 or self.proposal_events < 0:
            raise ValueError("adaptive counters must be >= 0")

    @property
    def selected_k(self) -> int:
        return self.levels[self.level_index]


def update_adaptive_draft_state(
    state: AdaptiveDraftState,
    proposed: int,
    accepted: int,
) -> dict[str, int | float | str | bool | list[int]]:
    if proposed <= 0:
        raise ValueError("proposed must be > 0")
    if accepted < 0 or accepted > proposed:
        raise ValueError("accepted must be within [0, proposed]")

    selected_k_before = state.selected_k
    ema_before = state.acceptance_ema
    streak_before = state.full_accept_streak
    event_acceptance = accepted / proposed
    state.acceptance_ema = 0.5 * event_acceptance + 0.5 * state.acceptance_ema
    state.proposal_events += 1
    reason = "hold"

    if accepted == 0:
        state.full_accept_streak = 0
        state.level_index = 0
        reason = "zero_accept"
    elif event_acceptance < 0.5 or state.acceptance_ema < 0.5:
        state.full_accept_streak = 0
        state.level_index = max(0, state.level_index - 1)
        reason = "weak_acceptance"
    elif accepted == proposed:
        state.full_accept_streak += 1
        reason = "full_accept_streak"
        if state.acceptance_ema >= 0.75 and state.full_accept_streak >= 2:
            state.level_index = min(len(state.levels) - 1, state.level_index + 1)
            state.full_accept_streak = 0
            reason = "promote"
    else:
        state.full_accept_streak = 0

    selected_k_after = state.selected_k
    return {
        "levels": list(state.levels),
        "proposal_event": state.proposal_events,
        "proposed_tokens": proposed,
        "accepted_tokens": accepted,
        "event_acceptance": event_acceptance,
        "acceptance_ema_before": ema_before,
        "acceptance_ema_after": state.acceptance_ema,
        "full_accept_streak_before": streak_before,
        "full_accept_streak_after": state.full_accept_streak,
        "selected_k_before": selected_k_before,
        "selected_k_after": selected_k_after,
        "transition_reason": reason,
        "promoted": selected_k_after > selected_k_before,
        "demoted": selected_k_after < selected_k_before,
    }
```

- [ ] **Step 4: Run the dependency-light test suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
```

Expected final line: `ngram speculative tests passed`.

- [ ] **Step 5: Commit the state machine**

```bash
git add tinyvllm/speculative/ngram.py tools/test_ngram_speculative.py
git commit -m "Add adaptive ngram draft policy"
```

---

### Task 2: Profiler Policy Dispatch and Event Accounting

**Files:**
- Modify: `tools/profile_ngram_commit.py`
- Modify: `tools/test_ngram_speculative.py`

**Interfaces:**
- Consumes: `AdaptiveDraftState`, `update_adaptive_draft_state`, and existing `verify_and_commit_block(...)`.
- Produces: CLI `--draft-policy {fixed,adaptive}` with default `fixed`.
- Produces: `validate_profile_args(args) -> None` before model loading.
- Produces: `propose_draft(history: list[int], args, max_draft_tokens: int | None = None) -> DraftProposal`.
- Candidate-only adaptive events add `adaptive_transition`; fixed events add `selected_k` but never mutate adaptive state.

- [ ] **Step 1: Write failing profiler validation and dispatch tests**

Add these aliases and tests:

```python
validate_profile_args = profile_ngram.validate_profile_args


def test_propose_draft_accepts_per_event_cap_without_mutating_args():
    class Args:
        draft_source = "ngram"
        ngram_size = 2
        max_draft_tokens = 4

    draft = propose_draft([1, 2, 3, 4, 1, 2], Args(), max_draft_tokens=1)
    assert draft.tokens == [3]
    assert Args.max_draft_tokens == 4


def test_profile_validation_rejects_adaptive_non_ngram_source():
    class Args:
        model = "model"
        temperature = 0.0
        max_commit_events = 0
        warmup_output_len = 1
        simulate_kv_upload_mb = 0.0
        draft_policy = "adaptive"
        draft_source = "dflash-toy"
        mode = "candidate-only"
        max_num_seqs = 1
        max_draft_tokens = 4

    try:
        validate_profile_args(Args())
    except ValueError as exc:
        assert "adaptive draft policy requires --draft-source ngram" in str(exc)
    else:
        raise AssertionError("adaptive non-ngram source accepted")


def test_profile_validation_requires_single_sequence_for_adaptive():
    class Args:
        model = "model"
        temperature = 0.0
        max_commit_events = 0
        warmup_output_len = 1
        simulate_kv_upload_mb = 0.0
        draft_policy = "adaptive"
        draft_source = "ngram"
        mode = "candidate-only"
        max_num_seqs = 2
        max_draft_tokens = 4

    try:
        validate_profile_args(Args())
    except ValueError as exc:
        assert "--max-num-seqs 1" in str(exc)
    else:
        raise AssertionError("batched adaptive profile accepted")
```

- [ ] **Step 2: Run tests and confirm the new profiler interface fails**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
```

Expected: failure because `validate_profile_args` and the per-event proposal cap do not exist.

- [ ] **Step 3: Add CLI validation and per-event proposal cap**

In `parse_args()` add:

```python
p.add_argument(
    "--draft-policy",
    choices=["fixed", "adaptive"],
    default="fixed",
    help="Use a fixed proposal cap or the profiler-only adaptive n-gram cap.",
)
```

Import the adaptive interfaces beside the current aliases:

```python
AdaptiveDraftState = ngram.AdaptiveDraftState
update_adaptive_draft_state = ngram.update_adaptive_draft_state
```

Change proposal dispatch to:

```python
def propose_draft(history: list[int], args, max_draft_tokens: int | None = None) -> DraftProposal:
    draft_cap = args.max_draft_tokens if max_draft_tokens is None else int(max_draft_tokens)
    if args.draft_source == "ngram":
        draft = propose_ngram_draft(history, args.ngram_size, draft_cap)
        return DraftProposal(
            tokens=list(draft.tokens),
            source="ngram",
            metadata={
                "match_start": draft.match_start,
                "ngram_size": draft.ngram_size,
                "selected_k": draft_cap,
            },
        )
```

Use `draft_cap` in both toy-source branches too, preserving fixed behavior when no override is supplied.

Extract argument checks from `run_profile()` into:

```python
def validate_profile_args(args) -> None:
    if args.model is None:
        raise ValueError("--model is required unless a synthetic smoke flag is set")
    if args.temperature != 0.0:
        raise ValueError("S3/S4 profiler currently supports greedy decoding only (--temperature 0.0)")
    if args.max_commit_events < 0:
        raise ValueError("--max-commit-events must be >= 0; use 0 for unlimited")
    if args.warmup_output_len < 0:
        raise ValueError("--warmup-output-len must be >= 0")
    if args.simulate_kv_upload_mb < 0:
        raise ValueError("--simulate-kv-upload-mb must be >= 0")
    if args.max_draft_tokens <= 0:
        raise ValueError("--max-draft-tokens must be > 0")
    if args.draft_policy == "adaptive":
        if args.draft_source != "ngram":
            raise ValueError("adaptive draft policy requires --draft-source ngram")
        if args.mode != "candidate-only":
            raise ValueError("adaptive draft policy requires --mode candidate-only")
        if args.max_num_seqs != 1:
            raise ValueError("adaptive draft policy requires --max-num-seqs 1")
```

Call this helper in `run_profile()` after synthetic-smoke early returns and before `_create_llm()`.

- [ ] **Step 4: Integrate one adaptive state per candidate sequence**

When constructing `stats_by_candidate` in `run_candidate_only_profile()`, add:

```python
"adaptive_state": AdaptiveDraftState() if args.draft_policy == "adaptive" else None,
```

Before each proposal:

```python
adaptive_state = stats["adaptive_state"]
selected_k = adaptive_state.selected_k if adaptive_state is not None else args.max_draft_tokens
draft = propose_draft(candidate.token_ids, args, max_draft_tokens=selected_k)
```

After a non-empty proposal is verified, add:

```python
event["selected_k"] = selected_k
event["proposed_tokens"] = len(draft.tokens)
event["wasted_draft_tokens"] = len(draft.tokens) - event["accepted_count"]
if adaptive_state is not None:
    event["adaptive_transition"] = update_adaptive_draft_state(
        adaptive_state,
        proposed=len(draft.tokens),
        accepted=event["accepted_count"],
    )
```

Do not call `update_adaptive_draft_state()` when `draft.tokens` is empty. Add to each candidate summary:

```python
"draft_policy": args.draft_policy,
"selected_k_counts": {
    str(level): sum(
        1 for event in stats["verify_events"]
        if event.get("selected_k") == level
    )
    for level in (1, 2, 4)
},
"adaptive_final_state": (
    {
        "selected_k": stats["adaptive_state"].selected_k,
        "acceptance_ema": stats["adaptive_state"].acceptance_ema,
        "full_accept_streak": stats["adaptive_state"].full_accept_streak,
        "proposal_events": stats["adaptive_state"].proposal_events,
    }
    if stats["adaptive_state"] is not None else None
),
```

Add aggregate fields:

```python
wasted_draft_tokens = drafted_tokens - accepted_tokens
zero_accept_verify_ms = sum(
    float(event.get("timing_ms", {}).get("verify_commit_total_ms", 0.0))
    for event in result_verify_events
    if event["accepted_count"] == 0
)
```

and report `wasted_draft_tokens`, `draft_waste_rate`, `zero_accept_event_rate`, and `zero_accept_verify_ms`.

- [ ] **Step 5: Add an adaptive event-shape helper test**

Factor event decoration into:

```python
def attach_draft_policy_event(
    event: dict,
    draft: DraftProposal,
    selected_k: int,
    adaptive_state: AdaptiveDraftState | None,
) -> dict:
    ...
```

Then test it without Torch:

```python
def test_attach_draft_policy_event_updates_adaptive_after_verification():
    state = AdaptiveDraftState()
    event = profile_ngram.attach_draft_policy_event(
        {"accepted_count": 0, "timing_ms": {"verify_commit_total_ms": 3.5}},
        profile_ngram.DraftProposal(tokens=[10, 11], source="ngram"),
        selected_k=2,
        adaptive_state=state,
    )
    assert event["selected_k"] == 2
    assert event["proposed_tokens"] == 2
    assert event["wasted_draft_tokens"] == 2
    assert event["adaptive_transition"]["selected_k_after"] == 1
    assert state.selected_k == 1
```

- [ ] **Step 6: Run local profiler tests and compilation**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_adaptive_pycache \
  python3 -m py_compile \
  tinyvllm/speculative/ngram.py \
  tools/profile_ngram_commit.py \
  tools/test_ngram_speculative.py
```

Expected: tests end with `ngram speculative tests passed`; compilation exits 0.

- [ ] **Step 7: Commit profiler integration**

```bash
git add tools/profile_ngram_commit.py tools/test_ngram_speculative.py
git commit -m "Integrate adaptive ngram profiling"
```

---

### Task 3: Deterministic Gate, Artifact Verifier, and Synthetic Decision Tests

**Files:**
- Create: `tools/adaptive_ngram_gate.py`
- Create: `tools/test_adaptive_ngram_gate.py`

**Interfaces:**
- Produces: committed `PROMPT_BANK` with names `natural_prose`, `structured_mixed`, `repeated_long_context`, and `transition_heavy`.
- Produces: `build_run_specs(repetitions: int, base_seed: int) -> list[dict]`.
- Produces: `replay_adaptive_trajectory(events: list[dict]) -> dict`.
- Produces: `summarize_rows(manifest: dict, raw_rows: list[dict], event_rows: list[dict]) -> dict`.
- Produces: `verify_artifacts(out_dir: pathlib.Path) -> dict`.
- CLI subcommands: `run`, `summarize`, and `verify`.

- [ ] **Step 1: Write failing prompt-bank, matrix, trajectory, threshold, and completeness tests**

Create `tools/test_adaptive_ngram_gate.py` with import-by-path matching existing tool tests and these focused cases:

```python
def test_prompt_bank_has_four_stable_single_sequence_classes():
    assert [item["name"] for item in gate.PROMPT_BANK] == [
        "natural_prose",
        "structured_mixed",
        "repeated_long_context",
        "transition_heavy",
    ]
    assert {item["workload_class"] for item in gate.PROMPT_BANK} == {
        "natural",
        "mixed",
        "high_repeat",
        "transition_heavy",
    }
    for item in gate.PROMPT_BANK:
        assert item["prompt"]
        assert item["max_output_len"] > 0
        assert item["prompt_sha256"] == gate.sha256_text(item["prompt"])


def test_build_run_specs_is_complete_unique_and_deterministic():
    first = gate.build_run_specs(repetitions=2, base_seed=20260714)
    second = gate.build_run_specs(repetitions=2, base_seed=20260714)
    assert first == second
    assert len(first) == 4 * 5 * 2
    keys = [item["run_key"] for item in first]
    assert len(keys) == len(set(keys))
    assert {item["policy"] for item in first} == {
        "baseline",
        "fixed_k1",
        "fixed_k2",
        "fixed_k4",
        "adaptive",
    }
    assert all(item["max_num_seqs"] == 1 for item in first)


def test_replay_adaptive_trajectory_detects_tampering():
    events = [
        synthetic_adaptive_event(2, 2, before=2, after=2, ema_before=0.5, ema_after=0.75, streak_before=0, streak_after=1),
        synthetic_adaptive_event(2, 2, before=2, after=4, ema_before=0.75, ema_after=0.875, streak_before=1, streak_after=0),
    ]
    assert gate.replay_adaptive_trajectory(events)["valid"] is True
    events[1]["adaptive_transition"]["selected_k_after"] = 1
    replay = gate.replay_adaptive_trajectory(events)
    assert replay["valid"] is False
    assert replay["fail_reasons"]


def test_summarize_rows_returns_go_for_committed_threshold_case():
    manifest, rows, events = synthetic_complete_gate_rows(
        repetitions=7,
        baseline_tps=100.0,
        fixed_tps={1: 103.0, 2: 104.0, 4: 105.0},
        adaptive_tps=108.0,
        adaptive_waste=20,
        fixed_k4_waste=40,
        adaptive_zero_ms=8.0,
        fixed_k4_zero_ms=12.0,
    )
    summary = gate.summarize_rows(manifest, rows, events)
    assert summary["decision"] == "GO"
    assert summary["correctness_pass"] is True
    assert summary["observed_rows"] == 140


def test_summarize_rows_near_best_requires_both_waste_reductions():
    manifest, rows, events = synthetic_complete_gate_rows(
        repetitions=7,
        baseline_tps=100.0,
        fixed_tps={1: 104.0, 2: 106.0, 4: 107.0},
        adaptive_tps=106.5,
        adaptive_waste=31,
        fixed_k4_waste=40,
        adaptive_zero_ms=11.0,
        fixed_k4_zero_ms=12.0,
    )
    summary = gate.summarize_rows(manifest, rows, events)
    assert summary["decision"] == "NO_GO"
    assert "adaptive_vs_fixed_gate_failed" in summary["decision_reasons"]


def test_summarize_rows_marks_missing_or_failed_process_incomplete():
    manifest, rows, events = synthetic_complete_gate_rows(repetitions=1)
    assert gate.summarize_rows(manifest, rows[:-1], events)["decision"] == "INCOMPLETE"
    rows[-1]["process"]["returncode"] = 1
    assert gate.summarize_rows(manifest, rows, events)["decision"] == "INCOMPLETE"


def test_natural_prompt_regression_forces_no_go():
    manifest, rows, events = synthetic_complete_gate_rows(repetitions=7)
    for row in rows:
        if row["prompt_name"] == "natural_prose" and row["policy"] == "adaptive":
            row["output_tokens_per_s"] = 94.0
    summary = gate.summarize_rows(manifest, rows, events)
    assert summary["decision"] == "NO_GO"
    assert "natural_or_transition_regression" in summary["decision_reasons"]
```

The test helper must emit all schema fields used by the real summarizer, including exact-output hashes, selected levels, transition records, process result, timings, and run keys; do not bypass validation with a special test-only summarizer branch.

- [ ] **Step 2: Run the gate tests and confirm the module is missing**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_adaptive_ngram_gate.py
```

Expected: import failure for `tools/adaptive_ngram_gate.py`.

- [ ] **Step 3: Implement the committed prompt bank and deterministic run matrix**

Use literal prompts and stable hashes:

```python
PROMPT_BANK_BASE = (
    {
        "name": "natural_prose",
        "workload_class": "natural",
        "prompt": (
            "Explain why a small engineering team should separate correctness "
            "benchmarks from performance benchmarks. Use concrete examples and "
            "finish with a short recommendation."
        ),
        "max_output_len": 96,
    },
    {
        "name": "structured_mixed",
        "workload_class": "mixed",
        "prompt": (
            "Continue this deterministic checklist with eight more items, keeping "
            "the exact format:\\n- check input\\n- check output\\n- record timing\\n"
        ),
        "max_output_len": 96,
    },
    {
        "name": "repeated_long_context",
        "workload_class": "high_repeat",
        "prompt": (
            "alpha beta gamma delta epsilon alpha beta gamma delta epsilon "
            * 96
            + "\\nContinue the sequence exactly:"
        ),
        "max_output_len": 128,
    },
    {
        "name": "transition_heavy",
        "workload_class": "transition_heavy",
        "prompt": (
            "A A A A B B B B C C C C. Now switch topics and explain in natural "
            "language why repeated patterns can stop abruptly, then emit: "
            "A A A A B B B B C C C C."
        ),
        "max_output_len": 112,
    },
)
PROMPT_BANK = tuple(
    {**item, "prompt_sha256": sha256_text(item["prompt"])}
    for item in PROMPT_BANK_BASE
)
```

Build each repetition's 20 run specs, shuffle them with `random.Random(base_seed + repetition)`, and assign `run_order` after shuffling. Policy arguments are:

```python
POLICIES = {
    "baseline": {"mode": "baseline-only", "draft_policy": "fixed", "max_draft_tokens": None},
    "fixed_k1": {"mode": "candidate-only", "draft_policy": "fixed", "max_draft_tokens": 1},
    "fixed_k2": {"mode": "candidate-only", "draft_policy": "fixed", "max_draft_tokens": 2},
    "fixed_k4": {"mode": "candidate-only", "draft_policy": "fixed", "max_draft_tokens": 4},
    "adaptive": {"mode": "candidate-only", "draft_policy": "adaptive", "max_draft_tokens": 4},
}
```

- [ ] **Step 4: Implement subprocess execution and canonical raw-row normalization**

For every run spec invoke one process:

```python
cmd = [
    python_bin,
    str(repo_root / "tools" / "profile_ngram_commit.py"),
    "--model", model_path,
    "--prompt", prompt["prompt"],
    "--max-output-len", str(prompt["max_output_len"]),
    "--warmup-output-len", str(min(8, prompt["max_output_len"])),
    "--temperature", "0.0",
    "--ngram-size", "5",
    "--max-commit-events", "0",
    "--mode", policy["mode"],
    "--draft-policy", policy["draft_policy"],
    "--max-num-seqs", "1",
    "--max-model-len", "4096",
    "--gpu-memory-utilization", "0.7",
    "--out-json", str(process_json),
]
```

Append `--max-draft-tokens` for fixed/adaptive candidates. For every process allocate two unused local TCP ports before launch, reject equality, and pass them through a copied environment as `TINYVLLM_DIST_PORT` and `MASTER_PORT`. Store the command, ports, stdout/stderr log paths, return code, start/end timestamps, and parsed profiler result.

Normalize baseline rows with output token IDs from `per_prompt[0]["token_ids"]`. Normalize candidate rows with the same field plus:

```python
{
    "proposal_events": summary["commit_attempts"],
    "no_draft_positions": summary["no_draft_steps"],
    "drafted_tokens": summary["drafted_tokens"],
    "accepted_tokens": summary["accepted_count"],
    "wasted_draft_tokens": summary["wasted_draft_tokens"],
    "draft_waste_rate": summary["draft_waste_rate"],
    "zero_accept_events": summary["zero_accept_events"],
    "zero_accept_event_rate": summary["zero_accept_event_rate"],
    "zero_accept_verify_ms": summary["zero_accept_verify_ms"],
    "verify_timing_ms": summary["verify_timing_ms"],
    "autoregressive_steps_avoided": summary["candidate_autoregressive_steps_avoided"],
}
```

Write `manifest.json` before running, append validated unique rows atomically through a temporary file plus `Path.replace()`, and preserve successful unique rows on `--resume`.

- [ ] **Step 5: Implement independent trajectory replay, aggregation, and fixed decision**

Trajectory replay must instantiate a fresh `AdaptiveDraftState`, confirm every event's `selected_k` equals the current state, recompute `update_adaptive_draft_state()` from raw proposed/accepted counts, and compare the complete transition record.

Aggregation must:

1. require exactly `len(PROMPT_BANK) * 5 * repetitions` unique rows;
2. require one baseline for every prompt/repetition candidate comparison;
3. compare candidate output token IDs exactly against that isolated baseline;
4. reject non-finite or non-positive elapsed/tps values;
5. require every process return code to be zero and profiler `gate_pass=true`;
6. sum output tokens and elapsed seconds across all four prompts per policy/repetition;
7. take the median of seven repetition-level aggregate throughputs;
8. report per-prompt seven-run medians;
9. sum waste and zero-accept verify milliseconds per policy/repetition, then use seven-run medians for policy comparison;
10. require adaptive exercise across the suite: non-empty proposals for each repeat-capable prompt, at least two selected levels, and at least one real promotion or demotion.

Use exactly:

```python
adaptive_vs_baseline = adaptive_tps / baseline_tps - 1.0
adaptive_vs_best_fixed = adaptive_tps / best_fixed_tps - 1.0
adaptive_waste_reduction_vs_k4 = 1.0 - adaptive_waste / fixed_k4_waste
adaptive_zero_cost_reduction_vs_k4 = 1.0 - adaptive_zero_ms / fixed_k4_zero_ms
```

The decision is `GO` only if correctness passes, adaptive exercise passes, `adaptive_vs_baseline >= 0.05`, and either:

```python
adaptive_vs_best_fixed >= 0.02
```

or all of:

```python
adaptive_vs_best_fixed >= -0.01
adaptive_waste_reduction_vs_k4 >= 0.20
adaptive_zero_cost_reduction_vs_k4 >= 0.15
```

and both natural/transition per-prompt ratios are at least `0.95`. Complete but threshold-missing evidence is `NO_GO`; structural/process failure is `INCOMPLETE`.

- [ ] **Step 6: Implement artifact verification and Markdown report generation**

`verify` must load only `manifest.json`, `raw_rows.json`, and `event_rows.json`, recompute the full summary, compare it structurally to `summary.json`, and regenerate the report text in memory to compare with `report.md`.

The report includes:

- decision and exact reasons;
- source commit, dirty state, model identifier/path, host, Python, GPU/environment fields;
- expected/observed row count;
- median aggregate throughput table;
- per-prompt median table;
- acceptance, draft waste, zero-accept cost, selected-`K`, and transitions;
- exact-output/trajectory audit status;
- fixed thresholds copied from the manifest;
- explicit single-sequence and Qwen3-0.6B-only claim boundaries;
- next direction for both `GO` and `NO_GO`.

- [ ] **Step 7: Run synthetic gate tests and compilation**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_adaptive_ngram_gate.py
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_adaptive_pycache \
  python3 -m py_compile \
  tools/adaptive_ngram_gate.py \
  tools/test_adaptive_ngram_gate.py
```

Expected final line: `adaptive ngram gate tests passed`; compilation exits 0.

- [ ] **Step 8: Commit the gate and verifier**

```bash
git add tools/adaptive_ngram_gate.py tools/test_adaptive_ngram_gate.py
git commit -m "Add adaptive ngram decision gate"
```

---

### Task 4: Isolated Remote Runner and One-Repetition Smoke

**Files:**
- Create: `tools/run_adaptive_ngram_gate_remote.sh`
- Modify: `tools/test_adaptive_ngram_gate.py`

**Interfaces:**
- Produces: shell commands `preflight`, `smoke`, and `canonical`.
- Uses: `REMOTE_HOST`, `REMOTE_PYTHON`, `REMOTE_BASE`, `CUDA_VISIBLE_DEVICES`, and optional `MODEL_PATH`.
- The remote runner creates a unique upload directory, discovers the model when `MODEL_PATH` is absent, runs the gate there, verifies remotely, downloads artifacts, and verifies again locally.

- [ ] **Step 1: Add a dependency-manifest test**

Expose in the gate module:

```python
REQUIRED_UPLOAD_PATHS = (
    "tinyvllm",
    "tools/draft_model_schema.py",
    "tools/profile_ngram_commit.py",
    "tools/adaptive_ngram_gate.py",
)
```

Test:

```python
def test_required_upload_paths_cover_profiler_imports():
    assert gate.REQUIRED_UPLOAD_PATHS == (
        "tinyvllm",
        "tools/draft_model_schema.py",
        "tools/profile_ngram_commit.py",
        "tools/adaptive_ngram_gate.py",
    )
```

- [ ] **Step 2: Write the fail-fast shell runner**

Create `tools/run_adaptive_ngram_gate_remote.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
REMOTE_BASE="${REMOTE_BASE:-/data00/home/sitian/sitian-workspace01/tllm/adaptive-ngram-gates}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-7}"
MODE="${1:-smoke}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)-$$}"
REMOTE_DIR="${REMOTE_BASE}/${RUN_TAG}"
LOCAL_OUT="${LOCAL_OUT:-${REPO_ROOT}/experiments/adaptive_ngram/${RUN_TAG}}"
```

Model discovery must run remotely and accept only a directory whose basename or resolved config identifies Qwen3-0.6B:

```bash
discover_model() {
  ssh "${REMOTE_HOST}" "${REMOTE_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

roots = [
    Path("/data00/home/sitian/sitian-workspace01/.ms_cache"),
    Path("/data00/home/sitian/sitian-workspace01"),
]
candidates = []
for root in roots:
    if not root.exists():
        continue
    for config in root.rglob("config.json"):
        text = str(config.parent).lower().replace("_", "").replace("-", "").replace(".", "")
        try:
            payload = json.loads(config.read_text())
        except Exception:
            continue
        model_type = str(payload.get("model_type", "")).lower()
        if "qwen3" in text and ("06b" in text or payload.get("hidden_size") == 1024) and model_type == "qwen3":
            candidates.append(str(config.parent.resolve()))
print(sorted(set(candidates))[0] if candidates else "")
PY
}
```

Fail if discovery returns empty or multiple ambiguous candidates without a deterministic recorded choice. Record the selected path in the local run log and gate manifest.

Upload only the current source snapshot:

```bash
ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_DIR}/tools'"
tar -C "${REPO_ROOT}" -cf - \
  tinyvllm \
  tools/draft_model_schema.py \
  tools/profile_ngram_commit.py \
  tools/adaptive_ngram_gate.py |
  ssh "${REMOTE_HOST}" "tar -C '${REMOTE_DIR}' -xf -"
```

Run remote preflight:

```bash
ssh "${REMOTE_HOST}" \
  "cd '${REMOTE_DIR}' && \
   PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${REMOTE_DIR}' \
   '${REMOTE_PYTHON}' -m py_compile \
     tinyvllm/speculative/ngram.py \
     tools/profile_ngram_commit.py \
     tools/adaptive_ngram_gate.py"
```

For `smoke`, pass `--repetitions 1`; for `canonical`, pass `--repetitions 7`. Always pass `CUDA_VISIBLE_DEVICES`, discovered model path, source commit, source dirty state, remote host, and Python into the gate manifest. Keep the remote directory after failure and print it.

- [ ] **Step 3: Validate shell syntax and local tests**

Run:

```bash
bash -n tools/run_adaptive_ngram_gate_remote.sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_adaptive_ngram_gate.py
```

Expected: shell exits 0; tests end with `adaptive ngram gate tests passed`.

- [ ] **Step 4: Commit the remote runner**

```bash
git add tools/run_adaptive_ngram_gate_remote.sh tools/test_adaptive_ngram_gate.py
git commit -m "Add isolated adaptive ngram remote runner"
```

- [ ] **Step 5: Run one-repetition remote smoke**

Run:

```bash
RUN_TAG="qwen3-06b-smoke-$(date +%Y%m%d-%H%M%S)" \
  tools/run_adaptive_ngram_gate_remote.sh smoke
```

Expected:

- model path is discovered and recorded;
- exactly `4 * 5 * 1 = 20` unique raw rows exist;
- every process has unique `TINYVLLM_DIST_PORT` and `MASTER_PORT`;
- every run has one prompt and `max_num_seqs=1`;
- every candidate output exactly matches its same-prompt baseline;
- adaptive events use only `1`, `2`, or `4`;
- remote and post-download `verify` both pass;
- decision may be `GO` or `NO_GO`, but must not be `INCOMPLETE`.

If the adaptive exercise condition is not reached in one repetition, the smoke may report a provisional `NO_GO`; that is acceptable only when row/process/correctness/artifact checks are complete.

- [ ] **Step 6: Record smoke defects before canonical execution**

If smoke reveals schema, prompt, timing, port, or process-lifecycle defects, add a failing local test reproducing each defect, implement the minimal correction, rerun local tests and smoke, then commit:

```bash
git add tools/adaptive_ngram_gate.py tools/test_adaptive_ngram_gate.py tools/run_adaptive_ngram_gate_remote.sh
git commit -m "Fix adaptive ngram gate smoke defects"
```

Do not change decision thresholds or substitute prompts based on favorable performance. Prompt changes are allowed only for objective gate-exercise defects and must be documented before the canonical run.

---

### Task 5: Seven-Repetition Canonical Gate and Persisted Decision

**Files:**
- Create: `experiments/adaptive_ngram/<resolved-model-and-date>/manifest.json`
- Create: `experiments/adaptive_ngram/<resolved-model-and-date>/raw_rows.json`
- Create: `experiments/adaptive_ngram/<resolved-model-and-date>/event_rows.json`
- Create: `experiments/adaptive_ngram/<resolved-model-and-date>/summary.json`
- Create: `experiments/adaptive_ngram/<resolved-model-and-date>/report.md`
- Modify: `README.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: the smoke-validated runner and unchanged committed thresholds.
- Produces: canonical `GO`, `NO_GO`, or `INCOMPLETE` decision with independently recomputable evidence.

- [ ] **Step 1: Run the seven-repetition canonical remote matrix**

Run from a clean committed worktree:

```bash
git status --short
RUN_TAG="qwen3-06b-canonical-$(date +%Y%m%d-%H%M%S)" \
  tools/run_adaptive_ngram_gate_remote.sh canonical
```

Expected: `4 * 5 * 7 = 140` unique rows, successful remote verification, successful download, and successful local verification. If interrupted, resume only with `--resume` against validated unique rows; never average duplicate or partial runs.

- [ ] **Step 2: Independently verify the downloaded artifact directory**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/adaptive_ngram_gate.py verify \
  --out-dir experiments/adaptive_ngram/<resolved-model-and-date>
```

Then inspect:

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("experiments/adaptive_ngram/<resolved-model-and-date>")
manifest = json.loads((root / "manifest.json").read_text())
rows = json.loads((root / "raw_rows.json").read_text())
events = json.loads((root / "event_rows.json").read_text())
summary = json.loads((root / "summary.json").read_text())

assert manifest["expected_rows"] == 140
assert len(rows) == 140
assert len({row["run_key"] for row in rows}) == 140
assert all(row["max_num_seqs"] == 1 for row in rows)
assert all(row["process"]["returncode"] == 0 for row in rows)
assert summary["observed_rows"] == 140
assert summary["decision"] in {"GO", "NO_GO"}
assert summary["correctness_pass"] is True
assert summary["trajectory_replay_pass"] is True
assert {event["selected_k"] for event in events if event["policy"] == "adaptive"} <= {1, 2, 4}
print("ADAPTIVE_NGRAM_CANONICAL_AUDIT_OK", summary["decision"])
PY
```

Expected: `ADAPTIVE_NGRAM_CANONICAL_AUDIT_OK GO` or `... NO_GO`. Any other result remains `INCOMPLETE`.

- [ ] **Step 3: Update README with measured, bounded usage and conclusion**

Add a concise section containing:

- the exact command `tools/run_adaptive_ngram_gate_remote.sh canonical`;
- artifact directory;
- normal/fixed/adaptive median aggregate throughput;
- output correctness status;
- acceptance/waste/zero-accept findings;
- final decision and fixed threshold reason;
- explicit statement that the result covers single-sequence greedy Qwen3-0.6B only and does not prove batch throughput, ragged verification, queueing latency, or memory reduction.

Do not replace the existing historical fixed n-gram result; distinguish it from this canonical adaptive gate.

- [ ] **Step 4: Append a detailed handoff entry**

Append to `AGENT_HANDOFF_STATE.md`:

- branch, source commit, remote host/Python/model path, GPU, run tag, and isolated remote directory;
- local and remote commands;
- smoke and canonical row counts;
- exact verifier outputs;
- throughput and waste comparison table;
- adaptive selected-`K` counts and transition exercise;
- decision with every passed/failed threshold;
- what the result proves and does not prove;
- next action:
  - on `GO`: design real ragged batched target verify plus load-aware `K=0..N`;
  - on `NO_GO`: retain policy/measurement code, prefer the best measured fixed policy only in its validated regime, and prioritize a higher-quality draft source;
  - on `INCOMPLETE`: list the concrete missing evidence and do not interpret performance.

- [ ] **Step 5: Run the full local verification set**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_adaptive_ngram_gate.py
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_adaptive_pycache \
  python3 -m py_compile \
  tinyvllm/speculative/ngram.py \
  tools/profile_ngram_commit.py \
  tools/adaptive_ngram_gate.py \
  tools/test_ngram_speculative.py \
  tools/test_adaptive_ngram_gate.py
bash -n tools/run_adaptive_ngram_gate_remote.sh
python3 tools/adaptive_ngram_gate.py verify \
  --out-dir experiments/adaptive_ngram/<resolved-model-and-date>
git diff --check
```

Expected: both test scripts pass, compilation and shell syntax exit 0, artifact verification passes, and `git diff --check` emits no output.

- [ ] **Step 6: Perform the prompt-to-artifact completion audit**

Create a checklist in the final handoff entry and verify each item against real evidence:

1. exact adaptive transition rules -> policy unit tests and event replay;
2. unchanged verify/commit path -> diff inspection plus exact output equality;
3. single sequence -> every raw command and row has one prompt and `max_num_seqs=1`;
4. five isolated policies -> 140 unique process rows;
5. four prompt classes -> manifest and per-prompt medians;
6. seven repetitions -> row keys and summary;
7. dynamic distinct ports -> per-process environment fields;
8. actual model discovery -> manifest path/config identity;
9. mandatory correctness -> all candidate/baseline token lists and trajectory replay;
10. fixed decision thresholds -> manifest values and independently recomputed summary;
11. canonical artifact set -> all five files;
12. remote and post-download verification -> saved command outputs;
13. README and handoff -> committed paths and result text;
14. claim boundaries -> report, README, and handoff.

Treat any missing or proxy-only evidence as incomplete and continue work.

- [ ] **Step 7: Commit canonical artifacts and documentation**

```bash
git add \
  experiments/adaptive_ngram/<resolved-model-and-date> \
  README.md \
  AGENT_HANDOFF_STATE.md
git commit -m "Record adaptive ngram speculation gate"
```

- [ ] **Step 8: Final clean-state audit**

Run:

```bash
git status --short --branch
git log --oneline -6
```

Expected: clean `feat/adaptive-ngram-speculation` worktree with separate commits for the approved design, implementation plan, policy, profiler integration, gate/runner, and canonical result.
