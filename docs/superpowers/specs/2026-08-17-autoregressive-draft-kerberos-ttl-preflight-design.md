# Autoregressive Draft Remote Gate Kerberos TTL Preflight Design

**Date:** 2026-08-17
**Status:** Approved design; implementation pending written-spec review

## Goal

Prevent a long source-bound TP4 CUDA Graph gate from starting when the local
Kerberos credential cannot remain valid for the expected campaign duration.
The guard must fail before creating a local run directory, staging remote
source, selecting GPUs for execution, or starting a remote worker.

The immediate failure being prevented occurred on August 17, 2026:

```text
run tag:
  20260817-steady-state-schema-v2-tp4-b4-q4-r1

progress before interruption:
  2 pair-level warmup pairs complete
  5 measured pairs complete
  pair 5 graph member in progress
  14 of 20 worker processes had produced JSON evidence

local Kerberos expiry:
  2026-08-17 21:32:00 CST

observed wrapper failure:
  SSH ControlMaster broken pipe
  remote CUDA graph gate failed with empty remote stderr
```

The interrupted tag remains incomplete environment evidence. It must not be
resumed, merged with a later campaign, or classified as a performance result.

## Selected Approach

Use a local fail-fast Kerberos lifetime guard before any run-side effects.

The runner will:

1. execute `klist --json` through an injectable command runner;
2. require the expected local principal;
3. locate a valid realm TGT;
4. parse its expiration timestamp;
5. require at least 90 minutes of remaining lifetime; and
6. return `INCONCLUSIVE_ENVIRONMENT` without contacting the remote host when
   the credential is absent, malformed, expired, or too close to expiration.

Ninety minutes is intentionally conservative. The interrupted campaign had
already consumed enough wall time to finish only 14 of 20 fresh workers, and
the complete workflow also includes preflight fingerprinting, source staging,
remote verification, checksumming, download, and local verification.

## Considered Alternatives

### Run the remote gate detached from SSH

Rejected for this gate. A detached process could survive an authentication
failure, but it weakens foreground ownership, makes precise process cleanup
harder, and can leave an unobserved GPU campaign running after the local
controller has lost authority. The current safety boundary requires the
foreground controller to own the run.

### Resume from completed worker JSON files

Rejected for controlled performance classification. Resuming after a long
authentication or environment gap would combine measurements from different
thermal, load, GPU-occupancy, and process-order windows. It would also weaken
the balanced eager-first/graph-first schedule. An interrupted tag remains
incomplete; the next attempt uses a new tag and starts from pair-level warmup
zero.

### Rely only on an SSH connectivity probe

Rejected as insufficient. A probe proves that the ticket works at one instant,
not that it can cover a long campaign. The August 17 failure began with a
working route and failed after most of the campaign had already run.

## Architecture

### Credential inspection

Add a pure parser for the JSON emitted by `klist --json`. It normalizes:

```text
cache
client principal
TGT principal
TGT expiration
remaining lifetime in seconds
```

The expected identities are:

```text
client principal:
  sitian@BYTEDANCE.COM

TGT principal:
  krbtgt/BYTEDANCE.COM@BYTEDANCE.COM
```

Timestamp parsing uses the compact Kerberos form:

```text
YYYYMMDDHHMMSS
```

The current wall clock is injected into the pure classification function so
expiration and threshold behavior are deterministic in tests.

### Preflight classification

The local classification returns one of:

```text
READY
INCONCLUSIVE_ENVIRONMENT
```

`READY` includes:

```text
principal
tgt_principal
expires_at
remaining_lifetime_seconds
minimum_required_lifetime_seconds
```

`INCONCLUSIVE_ENVIRONMENT` includes the same available metadata plus one
stable reason:

```text
local Kerberos cache is unavailable
local Kerberos payload is invalid
local Kerberos principal is unexpected
local Kerberos TGT is missing
local Kerberos TGT is expired
local Kerberos TGT lifetime is insufficient
```

No password, credential bytes, cache contents, or authentication secret may be
written into an artifact.

### Runner ordering

`execute_remote_gate()` performs operations in this order:

```text
classify local Kerberos lifetime
  -> if not READY, return INCONCLUSIVE_ENVIRONMENT
  -> create the new local run directory
  -> query remote GPUs and prerequisites
  -> build and stage the source-bound bundle
  -> execute the foreground gate
  -> run archived remote verifier
  -> build and check checksum manifest
  -> download artifacts
  -> run local verifier
```

The local Kerberos result is included in `preflight.json` only after the local
run directory is legitimately created for a READY attempt. A rejected attempt
does not reserve a run tag or create a partial artifact directory.

`--preflight-only` runs the same local check first. It does not attempt SSH
when local authentication cannot cover the minimum duration.

## Failure Semantics

The guard never:

- runs `kinit`;
- prompts for a password;
- reads the macOS Keychain;
- refreshes or changes a credential cache;
- kills an SSH, GPU, or remote worker process;
- deletes an interrupted run;
- treats missing authentication as a CUDA Graph failure; or
- converts partial worker files into a performance result.

If the local command itself fails, produces invalid JSON, or lacks the expected
TGT, the result is environment-inconclusive. The error message tells the user
to renew Kerberos outside the runner and retry with a new tag.

## Test Strategy

Add focused tests in
`tools/test_autoregressive_draft_cuda_graph_gate.py` for:

1. a valid expected TGT with more than 90 minutes remaining;
2. an expired TGT;
3. a valid TGT with less than 90 minutes remaining;
4. a missing TGT;
5. an unexpected client principal;
6. malformed JSON or malformed timestamp;
7. a failed `klist` subprocess;
8. `--preflight-only` avoiding all SSH commands when local auth fails;
9. `execute_remote_gate()` avoiding local directory creation and all remote
   commands when local auth fails; and
10. a READY local classification flowing into the existing remote preflight
    without changing TP4/B4/Q4 gate arguments.

The production implementation is written only after the new focused tests
demonstrate the expected RED state.

## Acceptance Criteria

The change is complete only when:

- the local guard rejects an expired or short-lived ticket before side effects;
- a valid long-lived ticket preserves the existing remote workflow;
- no automatic credential mutation is introduced;
- the existing schema-v2 gate/runtime suite remains green;
- compileall and `git diff --check` pass;
- the design, implementation plan, tests, implementation, audit, and handoff
  are committed and pushed; and
- after a real Kerberos renewal, a completely new source-bound tag finishes
  two warmup pairs, eight measured balanced pairs, both verifier receipts, and
  a valid checksum manifest.

The fail-fast guard improves campaign validity and operational safety. It does
not itself establish CUDA Graph correctness or performance.
