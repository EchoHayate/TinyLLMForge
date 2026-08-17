# Qwen3.5 TP4 Root-Logit Remote Runner Design

## Goal

Provide one dependency-light local CLI that safely resumes the frozen TP4
real root-logit correctness gate when four remote GPUs become truly idle.

The runner only orchestrates the already frozen source bundle and independent
verifier. It does not modify the gate, checkpoint, model, remote services, or
GPU resource policy.

## Fixed Identities

```text
remote target:
  sitian@10.232.195.203
remote Python:
  /data00/home/sitian/sitian-workspace01/tllm/env/bin/python
remote gate root:
  /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests
frozen source tag:
  qwen35-tp4-source-prep-20260729-010400
frozen source tree:
  b2d0b77de953e273dbf62f0e7b2bbe689ef33c183edf65830940e43123bb485f
exact artifact inventory:
  tp4_real_root_logit_correctness.json
  reference_logits.pt
  native_rank0_logits.pt
  rank_evidence.json
  source_manifest.json
```

## Modes

- `preflight`: invoke the frozen production GPU query and selector. It writes
  one local JSON result and returns non-zero unless four eligible GPUs exist.
- `native-smoke`: execute `preflight -> native-only distributed smoke`. READY
  launches the four frozen production native-rank workers without starting the
  official reference process. It validates rank launch identity, full rank-0
  logits, exact non-root `None`, topology, state mutation, collectives, process
  exit, process-group destruction, CUDA synchronization, cache emptying, and
  pool cleanup. It publishes exactly:
  `native_smoke.json`, `native_rank0_logits.pt`, and `rank_evidence.json`.
  BLOCKED returns exit code `2` and does not create remote work or publication
  paths.
- `run`: require a new safe run tag and successful preflight, invoke the frozen
  production `run` CLI, then require the remote run directory to contain
  exactly the five canonical artifacts.
- `download-only`: require an existing local destination with no canonical
  files, fetch exactly the five artifacts through a tar stream, and reject
  extra or missing remote files.
- `verify-only`: run the local independent verifier against the downloaded
  exact-five directory, binding it to the local frozen source root.
- `authority`: execute `preflight -> run -> download-only -> verify-only`.
  If preflight is blocked, print the same structured preflight JSON, return
  exit code `2`, and do not call any later phase.

## Safety

- Run tags match `[A-Za-z0-9_-]+` and must not already exist remotely or
  locally.
- SSH is non-interactive and uses Kerberos from the caller environment.
- The runner never uses `kill`, `pkill`, `rm -rf`, `git checkout`, `git reset`,
  `git clean`, or remote source mutation.
- Resource failure is terminal and must occur before reference/native workers.
- Native smoke never runs the reference worker and its three-file result is not
  accepted as an authoritative exact-five artifact.
- Failed and superseded remote runs remain preserved.
- Native child cleanup is bounded and applies only to processes constructed by
  the gate. On startup or runtime failure it performs
  `terminate -> bounded join -> kill -> bounded join`, then fails closed if a
  child is still alive. It never signals pre-existing remote processes.
- The four native rank joins share one monotonic deadline. The configured
  1800-second runtime timeout is a group budget, not four independent
  sequential 1800-second waits.
- Downloads use one tar stream and validate safe relative paths before
  extraction.
- `download-only` requires exactly five top-level remote entries and exactly
  five top-level regular files, independently rejecting extra directories,
  links, sockets, or files.
- Independent verification imports only the frozen verifier and reads the
  downloaded artifacts plus frozen local source root.

## Success Criteria

1. Dependency-light tests cover safe tags, exact-five inventory, command
   construction, preflight classification, download path safety, and authority
   ordering.
2. `preflight` against the current busy server fails closed and leaves only
   local preflight evidence.
3. `authority` against the current busy server returns exit code `2` without a
   traceback, leaves only local preflight evidence, and creates no remote
   run/work directory.
4. A clearly synthetic exact-five fixture can traverse the real
   `download-only -> verify-only` chain and pass the frozen independent
   verifier. This proves runner transport and verification wiring only.
5. A remote fixture with the exact five files plus one extra directory is
   rejected before any tar payload is returned.
6. No correctness, latency, throughput, cache, memory, compression, or quality
   claim is added by this runner.
7. `native-smoke` has dependency-light ordering/source-bound tests and a live
   BLOCKED test proving exit code `2`, exactly one local preflight file, and no
   remote smoke/work/publish directory.
