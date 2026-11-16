# Tokenizer Owned Child Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Terminate and reap one exclusively transferred direct child without raw-PID cleanup.

**Architecture:** One main-thread ownership record uses exact Popen handoff,
waitpid WNOHANG and bounded TERM/KILL escalation on the existing cleanup
window. No process creation or tree/remote cleanup API.

**Tech Stack:** Python 3.12 POSIX standard library and unittest.

## Global Constraints

- Existing checkout only; no branch/worktree, commit or push.
- No network/SSH, installation, telemetry, real runtime metadata discovery,
  tokenizer/model loading, GPU, training or candidate/task-source execution.
- Stdlib tests may create only their own fixed inert Python children with
  `-I -S`, empty environment, closed inherited FDs and no shell. Each fixture
  self-exits within 5 seconds, has bounded handshake/wait/cleanup, and creates
  no descendants. At most two such children live simultaneously.
- Preserve all accepted sources/tests/data/documents and sealed evidence.
- Use `apply_patch`; no inline code comments or alphabetic one-letter names.
- Production code never launches a process, signals a group, scans processes,
  reads/writes payload pipes, closes caller FDs, deletes paths or installs
  handlers. It signals only the exclusively transferred direct child.
- One attach attempt per instance and at most one termination attempt, one
  SIGTERM attempt and one SIGKILL attempt. No PID/PGID adoption API.
- Use exact `LaunchWindow` in CLEANUP/CLOSED; no new window, connection/audit
  claim, begin_cleanup or close. TERM grace is at most 1,000,000,000 ns;
  each sleep is at most 0.01 seconds and current cleanup time remaining.
- All ten enforcement/admission flags remain false; `cleanup_status` remains
  `CLEANUP_UNVERIFIED`, even when this one child has been reaped.

### Task 1: Exclusive direct-child lifecycle

**Files:**
- Create `experiments/lora_necessity/tokenizer_owned_child.py`.
- Create `experiments/lora_necessity/test_tokenizer_owned_child.py`.

**Requirements:** Read the complete binding design
`docs/superpowers/specs/2026-11-13-lora-tokenizer-owned-child-design.md`.
The exact state, ownership, observation, failure and cancellation semantics
are defined there. Preserve all sealed dependencies unchanged.

**Interfaces:** `OwnedChild()`, `attach(process)`, `poll()`,
`terminate(window)`, `snapshot()`, `OwnedChildError(ValueError)`.
Consume exact Popen and sealed LaunchWindow via its public methods.

- [x] Write failing construction/ownership tests using an explicit module
  assertion before implementation:

  ```python
  self.assertIsNotNone(importlib.util.find_spec("tokenizer_owned_child"))
  owner = importlib.import_module("tokenizer_owned_child").OwnedChild()
  self.assertEqual(owner.snapshot()["state"], "NEW")
  self.assertFalse(owner.snapshot()["child_reaped"])
  ```

  Use fixture-owned isolated Python children only, with bounded readiness and
  cleanup on failed assertions. Natural-exit test must obtain the real kernel
  status and synchronize Popen.returncode, not mock exit success.
- [x] Run focused RED and retain numbered exclusive logs plus receipts:
  `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -m unittest discover -s experiments/lora_necessity -p 'test_tokenizer_owned_child.py' -v`.
  Record exact command, exit code, log hash and source hashes before/after.
- [x] Implement the minimal tested constructor/attach/poll path, then GREEN.
  Fixed error skeleton:

  ```python
  class OwnedChildError(ValueError):
      def __init__(self):
          super().__init__("tokenizer owned child invalid")
  ```

  Preserve strong object/private PID identity, one weak-set handoff, default
  SIGCHLD and exact waitpid status checks. Reaped is based only on waitpid.
- [x] Expand RED/GREEN for bounded termination and all design validation
  cases. Prove actual TERM and TERM-ignore escalation, sibling preservation,
  ownership loss before signals, deadline/cancellation emergency behavior,
  no repeated signals and all public-state privacy/invariant requirements.
  Do not shrink the production grace; deterministic clock tests complement
  the real one-second TERM-ignore test. No fixture program exceeds 5 seconds.
- [x] Run focused final GREEN and one full suite:
  `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -m unittest discover -s experiments/lora_necessity -p 'test_*.py' -v`.
  All previous 277 unique test IDs must remain; final output pristine.
- [x] Self-review and write
  `/tmp/tlf-tokenizer-child-20261113-AE4kS9/task-1-report.md`
  with exact results, real/synthetic distinctions, source hashes, file scope,
  TDD limitations and any remaining concerns. No commit.
- [x] Main packages full additions and obtains independent task review.

## Main closeout

- [x] Extend complete previous protection manifest and record Git identity.
- [x] Independently run regression, static/import checks and package binding.
- [x] Obtain fresh whole-component review and generate local reports.
- [x] Reverify preservation and exclusively seal current evidence.

Completion is bound to the successfully generated and hash-verified
`/tmp/tlf-tokenizer-child-20261113-AE4kS9/final-verification.json`, not to
these checkboxes alone. Main independently passed 314 tests (277 preserved,
37 new); both review gates returned PASS / APPROVED with no findings.
No commit, push, real producer launch or scientific admission is included.

## Plan check

One cohesive source/test pair. Local test child creation does not authorize
real producer launch. Existing reader/window implementations stay sealed.
No group kill, resource-backend assumption or false cleanup/admission flag.
