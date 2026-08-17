# Qwen3.5 Checkpoint Candidate Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve verified model-manifest SHA256 across streaming/tiled checkpoint candidates and owner publication.

**Architecture:** Every loader validates a required SHA256 before I/O and stores it on its frozen candidate. The one-shot publication slot atomically stores owner plus fingerprint.

**Tech Stack:** Python dataclasses, existing checkpoint loaders and publication slot, dependency-light synthetic safetensors tests.

## Global Constraints

- Adaptive-ngram worktree only.
- Fingerprint is required lowercase SHA256.
- Validation must occur before shard open.
- Do not mutate the frozen model owner graph.
- No subagents, staging, commits, or evidence cleanup.
- Do not claim production memory or speed benefit.

---

### Task 1: RED/GREEN Streaming Provenance

- [x] Add valid retention and invalid pre-open rejection tests.
- [x] Add required loader keyword to test call sites.
- [x] Implement candidate field and loader validation.
- [x] Confirm streaming tests GREEN.

### Task 2: RED/GREEN Tiled Provenance

- [x] Add tiled and policy-tiled retention assertions.
- [x] Add invalid pre-open rejection tests.
- [x] Thread fingerprint through tile-plan loader.
- [x] Confirm tiled suites GREEN.

### Task 3: RED/GREEN Publication Slot Provenance

- [x] Add default-off owner/fingerprint assertions.
- [x] Add successful atomic publication assertions.
- [x] Add failure/replacement preservation assertions.
- [x] Implement slot fingerprint state.
- [x] Confirm publication tests GREEN.

### Task 4: Regression and Handoff

- [x] Run checkpoint loading, owner binding, identity, publication, and
  Scheduler integration suites.
- [x] Run compile, diff-check, staged-file, and runtime wiring audits.
- [x] Record provenance proof and worker handoff gap.
