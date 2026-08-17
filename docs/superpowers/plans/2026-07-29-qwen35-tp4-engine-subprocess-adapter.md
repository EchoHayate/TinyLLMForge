# Qwen3.5 TP4 Engine Subprocess Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a CPU-testable, non-automatic local subprocess adapter for the verified and single-use-authorized TP4 Engine remote execution plan.

**Architecture:** Preserve the process-free executor core and place all `subprocess.Popen` use in one isolated module implementing the existing injected runner protocol. The adapter has no CLI, validates executable/environment/output boundaries, captures bounded logs through files, and streams package stdout directly to a new binary file.

**Tech Stack:** Python 3 standard library, `subprocess.Popen`, temporary files, SHA-256, file-local tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, merge, create a branch, or create a PR.
- Do not execute SSH, `scp`, `nvidia-smi`, Torch, or a GPU workload.
- Preserve exact `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- Do not add a CLI, `main`, import-time execution, retries, or process killing.
- Keep the existing executor/plan/receipt/authorization modules free of subprocess execution.

---

### Task 1: Add the Isolated Adapter

**Files:**
- Create: `tools/qwen35_tp4_engine_remote_subprocess_adapter.py`
- Create: `tools/test_qwen35_tp4_engine_remote_subprocess_adapter.py`
- Modify: `tools/build_qwen35_tp4_engine_authority_configuration.py`
- Modify: `tools/qwen35_tp4_engine_remote_execution_plan.py`
- Modify: `tools/qwen35_tp4_engine_remote_execution_source_contract.py`
- Modify: `tools/test_qwen35_tp4_engine_remote_execution_source_contract.py`

**Interfaces:**
- Consumes: `command_runner(name, argv, stdout_path, env)` calls from `execute_plan`.
- Produces: `run_command(*, name, argv, stdout_path, env, popen_factory=subprocess.Popen) -> dict`.

- [x] **Step 1: Write failing adapter tests**

Require exact result dictionaries, allowlisted executables, `shell=False`,
bounded UTF-8 logs, binary package output identity, and no CLI surface.

- [x] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
python3 tools/test_qwen35_tp4_engine_remote_subprocess_adapter.py
```

Expected: failure because the adapter module does not exist.

- [x] **Step 3: Implement the minimal adapter**

Use temporary regular files for stdout/stderr, direct binary output for
package download, explicit argv/environment validation, `Popen(...,
shell=False)`, and independent SHA/size computation.

- [x] **Step 4: Extend immutable inventory and static safety checks**

Include the adapter in the Engine authority source bundle. Keep it excluded
from the process-free source set, but add AST checks proving no CLI,
`shell=True`, subprocess helpers other than `Popen`, or default executor
runner wiring.

- [x] **Step 5: Run focused and full CPU-safe gates**

Run the adapter tests, source-contract tests, the complete Engine authority
CPU-safe suite, focused `py_compile`, and `git diff --check`.

- [x] **Step 6: Update durable evidence**

Record exact test counts and preserve the boundary that no remote/GPU command
was executed and no correctness or performance benefit is yet claimable.

## Result

Implemented:

```text
tools/qwen35_tp4_engine_remote_subprocess_adapter.py
tools/test_qwen35_tp4_engine_remote_subprocess_adapter.py
```

The adapter is isolated from the process-free executor core and has no CLI or
automatic execution surface. It allowlists only `ssh`, `scp`, and the exact
local Python executable; requires the exact Kerberos cache binding; always
uses `shell=False`; captures bounded UTF-8 logs; streams package stdout to a
new binary file; independently reports package SHA/size; and deletes partial
package output on nonzero return, process-start exception, invalid logs, or
empty output.

The executor package-result protocol now distinguishes success from failure:
only a zero-return package result may carry and must validate output SHA/size.
A nonzero package result carries the ordinary three command-result fields and
is converted into bounded authorization-bound FAILED evidence.

Fresh validation:

```text
adapter focused tests:                 7 passed
executor focused tests:               11 passed
source-contract focused tests:         4 passed
remote-plan focused tests:             6 passed
configuration-builder focused tests:   4 passed
complete CPU-safe gate:              206 passed across 28 files
focused py_compile:                  passed
git diff --check:                    passed
staged files:                        0
```

No adapter call used the real `subprocess.Popen`; all process behavior was
tested through an injected fake factory. No SSH, `scp`, `nvidia-smi`, remote
directory, Torch load, or GPU workload was executed.
