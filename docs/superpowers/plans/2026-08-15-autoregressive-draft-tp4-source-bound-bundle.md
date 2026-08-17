# Autoregressive Draft TP4 Source-Bound Bundle Implementation Plan

> **For agentic workers:** Execute inline in the existing
> `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram` worktree. Do not create a
> worktree, dispatch subagents, stage, commit, push, stash, reset, or clean.

**Goal:** Publish a schema-v2 learned-drafter TP4 authority bundle whose
accepted-prefix evidence is reconstructable and whose verification runs from
deterministically archived source.

**Architecture:** Keep the inference runtime unchanged. Extend the existing
gate's evidence schema, add a standard-library verifier, and add a
temporary-directory bundle publisher that compares current-source and
archived-source receipts before exclusive publication.

**Tech Stack:** Python standard library, pytest, JSON, SHA-256, uncompressed
POSIX tar archives.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Keep `MAX_PROPOSAL_TOKENS=4`.
- Do not change verifier selection, fallback indexing, accepted-prefix
  semantics, target-KV transactions, Scheduler, n-gram, SAM, or MTP behavior.
- Do not run GPU, remote, NCCL, loaded-checkpoint, or performance workloads.
- Do not claim real movement, performance, promotion, or Phase 1 completion.
- Preserve the legacy single-JSON CLI.

---

### Task 1: Reconstructable Accepted-Prefix Rows

**Files:**
- Modify: `tools/autoregressive_draft_tp4_engine_gate.py`
- Modify: `tools/test_autoregressive_draft_tp4_engine_gate.py`

**Interfaces:**
- Produces: schema-v2 `acceptance_rows` with event, step, prompt, output
  boundary, proposal, and accepted-prefix identity.

- [x] Add failing tests for exact accepted-prefix fields and event ordering.
- [x] Run focused tests and confirm failure on missing schema-v2 fields.
- [x] Capture step-indexed acceptance identity without changing engine
  execution.
- [x] Strengthen `validate_gate_payload()` and set `SCHEMA_VERSION = 2`.
- [x] Run focused tests and confirm PASS.

### Task 2: Standard-Library Independent Verifier

**Files:**
- Create: `tools/verify_autoregressive_draft_tp4_engine_gate.py`
- Create: `tools/test_verify_autoregressive_draft_tp4_engine_gate.py`

**Interfaces:**
- Produces:
  `verify_run(run_dir: Path, source_root: Path) -> dict[str, object]`.
- Consumes: `result.json`, `source_manifest.json`, and an explicit archived
  source root.

- [x] Add failing tests for canonical result validation, manifest hashes,
  source drift, and bounded failures.
- [x] Run tests and confirm failure because the verifier is absent.
- [x] Implement standard-library module loading and canonical verification.
- [x] Run verifier tests and confirm PASS.

### Task 3: Deterministic Safe Source Archive

**Files:**
- Modify: `tools/autoregressive_draft_tp4_engine_gate.py`
- Modify: `tools/test_autoregressive_draft_tp4_engine_gate.py`
- Modify: `tools/test_verify_autoregressive_draft_tp4_engine_gate.py`

**Interfaces:**
- Produces: deterministic `source.tar`, per-file hashes, source-tree digest,
  and safe extraction into a fresh root.

- [x] Add failing tests for byte determinism and normalized tar metadata.
- [x] Add failing tests for traversal, links, special files, duplicates,
  unexpected/missing members, and payload hash mismatch.
- [x] Run tests and confirm expected failures.
- [x] Implement explicit inventory validation, deterministic tar creation,
  and manual safe extraction without `extractall()`.
- [x] Run archive/verifier tests and confirm PASS.

### Task 4: Atomic Authority Bundle Publication

**Files:**
- Modify: `tools/autoregressive_draft_tp4_engine_gate.py`
- Modify: `tools/test_autoregressive_draft_tp4_engine_gate.py`

**Interfaces:**
- Produces:
  `publish_authority_bundle(payload, output_dir, source_root) -> dict`.
- Adds CLI: mutually exclusive `--output` and `--output-dir`.

- [x] Add failing tests for current/archived receipt equality, atomic success,
  `.failed` preservation, and replacement refusal.
- [x] Run tests and confirm bundle publication is missing.
- [x] Implement temporary-directory assembly and exclusive publication.
- [x] Load the second verifier from safely extracted archived source.
- [x] Run focused tests and confirm PASS.

### Task 5: Regression and Audit Update

**Files:**
- Modify:
  `docs/superpowers/audits/2026-08-14-phase1-promotion-checklist.md`
- Modify:
  `docs/superpowers/audits/2026-08-15-phase1-prompt-to-artifact-coverage.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] Run gate, local validator, snapshot transport, verifier, and learned
  drafter dependency-light suites.
- [x] Run changed-file `py_compile`.
- [x] Run Markdown classification checks and scoped `git diff --check`.
- [x] Record source-bound bundle readiness while retaining loaded TP4,
  Proposal-KV movement, performance, Phase 1, and promotion as not
  established.

