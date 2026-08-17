# Qwen3.5 TP4 Root-Logit Remote Receipt Authority Design

## Goal

Add an immutable, single-use, receipt-bound execution authority around the
existing TP4 real root-logit remote runner without changing the frozen
correctness source or executing SSH/GPU work during implementation.

## Current Gap

`run_qwen35_tp4_real_root_logit_gate_remote.py` already implements:

```text
preflight -> run -> download -> verify
```

It enforces the exact remote target, frozen source-tree identity, four idle
GPUs, exact-five artifact inventory, safe download extraction, and independent
verification. However, the chain has no immutable plan, no single-use
authorization consumed before execution, and no receipt that binds all four
stages. Consequently the performance prerequisite provenance must currently
declare `root_logit_receipt_gap=true`.

## Approaches

### A. Semantic Four-Stage Protocol

Freeze the exact runner inputs and semantic stage identities into a plan.
Execute each stage through explicit injected callbacks, consume authorization
before the first callback, and publish a receipt that validates the stage
outputs and on-disk evidence.

Advantages:

- reuses the tested runner and frozen source;
- no new subprocess owner;
- small, independently testable modules;
- directly closes the prerequisite provenance gap.

Limitation:

- the receipt binds semantic stage inputs and outputs rather than every
  nested SSH subprocess argv.

### B. Command-Level Protocol Rewrite

Decompose the runner into every SSH/download/verify command and hash each
command in the receipt.

Advantages:

- strongest command-level audit.

Disadvantages:

- duplicates mature runner logic;
- large regression surface;
- requires a new stdin-aware adapter for streamed verification.

### C. Summary Receipt Around `execute_authority`

Call the existing all-in-one function and write one summary after success.

Advantages:

- minimal code.

Disadvantages:

- no stage prefix evidence;
- weak failure attribution;
- not independently replayable.

## Decision

Use Approach A. It closes the actual provenance gap with the least risk and
does not weaken any current safety gate. Command-level decomposition may be a
future hardening step, but it is not required to prove that one consumed
authorization produced one exact verified authority directory.

## Architecture

### Immutable Plan

Create:

```text
tools/qwen35_tp4_root_logit_remote_execution_plan.py
```

The plan binds:

- schema and run tag;
- exact target `sitian@10.232.195.203`;
- frozen source tag and source-tree SHA;
- canonical model-manifest SHA from the performance prerequisite contract;
- repository root and unique local run directory;
- exact stage order:
  `preflight`, `run`, `download`, `verify`;
- exact-five artifact names;
- minimum free bytes and active-compute-process prohibition;
- canonical semantic input for each stage;
- `execution_performed=false`.

The verifier requires the plan-local run directory not to exist when the plan
is built and rechecks every frozen constant against the production runner.

### Single-Use Authorization

Create:

```text
tools/qwen35_tp4_root_logit_remote_execution_authorization.py
```

The authorization binds the canonical plan SHA, run tag, frozen source SHA,
exact target, exact stage order, and a safe nonce. Consumption atomically
renames the active file before rewriting `consumed=true`.

### Execution Receipt

Create:

```text
tools/qwen35_tp4_root_logit_remote_execution_receipt.py
```

The receipt requires four ordered successful stage rows:

1. `preflight`
   - `status=READY`;
   - exact frozen source SHA;
   - four unique rank/GPU-index/GPU-UUID rows;
   - at least 24 GiB free per GPU;
   - no compute processes.
2. `run`
   - `status=REMOTE_PASS`;
   - exact run tag and remote run directory;
   - exact-five artifact names.
3. `download`
   - `status=DOWNLOADED`;
   - exact-five artifact names;
   - the plan-local `artifacts/` directory exists with only regular files.
4. `verify`
   - production independent verifier payload has `classification=PASS`;
   - exact case IDs, ranks `0..3`, and positive independent check count;
   - the on-disk `independent_verification.json` exactly equals the stage
     payload.

Receipt validation also reads the downloaded
`tp4_real_root_logit_correctness.json` and `source_manifest.json`. It invokes
the shared dependency-light `validate_authority_documents(...)` semantic
validator with the plan-bound frozen source SHA, and requires the source
manifest to carry that same source SHA. Model identity is therefore verified
from the authority document rather than invented as a field in the root
verifier's smaller return envelope.

The receipt binds the consumed authorization SHA/nonce and hashes every
canonical stage result. FAILED evidence contains only the completed stage
prefix and the next failed stage.

### Dependency-Injected Executor

Create:

```text
tools/qwen35_tp4_root_logit_remote_execution_executor.py
```

The executor:

- has no default stage runner;
- requires exact
  `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`;
- verifies the plan before consuming authorization;
- rejects pre-existing receipt, failure, consumed authorization, or local run
  targets;
- consumes authorization before the first callback;
- invokes explicit callbacks in frozen order;
- publishes PASS only after receipt validation;
- publishes bounded prefix-preserving FAILED evidence on error.

The production callback adapter is the existing runner API:

```text
execute_preflight
execute_run
execute_download
execute_verify
```

No frozen source file is modified.

## Prerequisite Adapter Integration

Update `qwen35_tp4_real_prerequisite_authority_adapter.py` so root-logit may
use the same receipt-bound provenance shape as cached continuation and Engine.
After a verified root-logit receipt chain exists:

```text
binding_kind = remote_execution_receipt
root_logit_receipt_gap = false
```

The v2 builder copies the root plan, consumed authorization, and receipt into
the self-contained prerequisite bundle. Legacy complete-directory-only input
must be rejected after this integration so the benchmark has one unambiguous
root authority path.

## Error Handling

- Any plan, authorization, stage result, evidence file, or identity mismatch
  fails closed.
- Authorization remains consumed after execution failure.
- Existing remote artifacts are never deleted or overwritten.
- Existing local run, receipt, failure, or consumed-authorization paths are
  never overwritten.
- FAILED logs and exception text are byte bounded.
- No PASS receipt is emitted for blocked preflight or verifier failure.

## Testing

Use file-local CPU-only tests with injected callbacks:

- plan schema, frozen constants, unique path, and tamper rejection;
- authorization identity, nonce safety, atomic single-use consumption;
- receipt validation for all four stages and exact on-disk evidence;
- blocked preflight, wrong GPU identity, extra artifact, changed verifier
  payload, source drift, and authorization drift;
- executor callback order, consume-before-first-callback, pre-existing target
  rejection, exact environment, and prefix-preserving failure;
- adapter-to-v2-bundle end-to-end test with root receipt provenance;
- AST contract proving new protocol modules import no subprocess API.

No SSH, `scp`, `nvidia-smi`, Torch/Transformers/CUDA initialization, model
load, Engine construction, or GPU workload is part of this implementation.

## Claim Boundary

This protocol closes an execution-provenance gap only. Until a real receipt
chain and all three real correctness authorities exist, no latency,
throughput, cache, GPU-memory, compression, quality, or accuracy gain is
claimable.
