# Tokenizer Discovery Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow dependency metadata queries without permitting backend loading, retain privacy-safe OPEN diagnostics, and perform one new tokenizer-only audit.

**Architecture:** Keep the import-blocking meta-path finder. Install an owned `importlib.util.find_spec` wrapper for prohibited names only; resolve actual specs through the remaining finders and replace their loader with a denying loader. Ordinary imports remain blocked at the first finder, including absent packages and stateful fallback finders. Diagnostics use fixed stage/error categories only.

**Tech Stack:** Python 3.12 stdlib, existing unittest suite, unchanged existing-host SSH/bootstrap.

## Global Constraints

- Keep the current experimental branch; no commit, push, worktree change, installation, training or generation.
- Preserve frozen 324-row input bytes, five tokenizer pins and all old attempts.
- One newly identified attempt after tests and independent review; no automatic retry.
- No private strings, paths, answers, token IDs, exception messages or traceback contents in diagnostics.
- Keep 4096 input / 256 target limits, stop-first behavior and all scientific flags.
- Ordinary deadlines and task/output protection remain. Do not reinstate removed aggregate resource quotas or automatic cleanup requirements.
- Evidence root: `/var/folders/0c/hl2r_50x28j2ysjvv2120t8h0000gn/T/tlf-tokenizer-discovery-20261113-pdcqnt4i`.

## Task 1: Discovery Versus Execution

Files: `experiments/lora_necessity/tokenizer_runtime_guard.py` and `test_tokenizer_runtime_guard.py`.

Interfaces: `RuntimeGuard.start/check/close` unchanged; public `importlib.util.find_spec(name, package=None)` retains presence/absence semantics. `RuntimeGuard.find_spec` continues to deny actual imports. Dotted queries that require importing a prohibited parent remain denied.

- [x] Add tests using a real temporary `safetensors.py`: discovery returns a spec without executing its body; `importlib.import_module("safetensors")` still fails and poisons the guard.
- [x] Cover absent results, later finders, namespace specs, direct returned-loader execution, wrapper replacement/restoration, relative names and prohibited-parent imports.
- [x] Run `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -m unittest discover -s experiments/lora_necessity -p test_tokenizer_runtime_guard.py -v`; save RED output.
- [x] Implement an owned discovery wrapper. For allowed names use the saved helper; for forbidden names resolve parent exactly as the public helper requires, iterate remaining finders once, copy the first real spec and attach a loader whose `create_module/exec_module` raise through the guard. Return `None` if absent. Never call the original backend loader.
- [x] Include wrapper identity in `check`; restore only the owned wrapper on close. Run the same tests and save GREEN output.

## Task 2: Sanitized OPEN Diagnostics

Files: `tokenizer_session.py`, `tokenizer_audit.py`, `tokenizer_audit_report.py` and their existing tests under `experiments/lora_necessity/`.

Interfaces: optional `SessionError.open_diagnostic`; optional trailing `build_report(..., open_diagnostic=None)`. Existing reports remain byte-shape compatible when the diagnostic is absent. When present, add `public.open_diagnostic` after strict validation.

- [x] Add failing tests for load/runtime/snapshot/observation stages, hostile exception text/type names, and actual controller propagation.
- [x] Stages: `GUARD_START`, `RUNTIME_METADATA`, `SNAPSHOT_VERIFY`, `TOKENIZER_LOAD`, `OBSERVE`, `FINAL_CHECK`. Error categories: exact builtin `ImportError`, `ModuleNotFoundError`, `ValueError`, `RuntimeError`, `OSError`, `TypeError`, `MemoryError`, otherwise `OTHER`. Include `guard_failed` as a boolean.
- [x] Keep `_raise_failure` generic and non-Exception propagation unchanged; construct diagnostics inside `open_session` from controlled stage plus exact builtin type, not exception text. Do not attach the original exception.
- [x] Controller passes diagnostics only for OPEN failures from `SessionError`. Report builder rejects unknown fields/categories or diagnostics on any other outcome.
- [x] Run all `test_tokenizer*.py` tests, save raw logs and counts. Preserve ordering, full denominators, stop and finalization behavior.

## Task 3: Review And One New Audit

Files: a new result document and append-only progress ledger; private run preparation/results in the evidence root.

- [x] Freeze revised source copies, a new UUID run ID/profile and unchanged inventory; build the existing launch manifest without opening SSH.
- [x] Have an independent reviewer inspect exact source changes, tests, manifest and scope; resolve blocking findings before execution.
- [x] Recheck local credentials. Exclusively record the attempt claim, then call existing `execute_once` with the reviewed manifest hash. Execution deviation: the first local call failed canonical-path validation before creating SSH; after preserving that rejection, the second local call made the sole remote connection.
- [x] Reconstruct the returned report through `build_report`, including optional OPEN diagnostics, without printing private payloads.
- [x] Record counts, actual stop/failure reason and residual-state uncertainty. Tokenizer feasibility is not LoRA benefit or RSI evidence.

## Execution Result

Tasks completed, but the scientific tokenizer check did not pass: run
`84533a4ce05649df8f4e94f826dbbbd6` returned OPEN_FAILED at TOKENIZER_LOAD with
ImportError and guard_failed=true; 0/324 measured. Final full local suite:
360 passed, 2 skipped. The reviewed compatibility P1 was fixed before the
single remote connection. No remote retry or training occurred.

See `experiments/lora_necessity/20261113-tokenizer-discovery-revision-results.md`.
