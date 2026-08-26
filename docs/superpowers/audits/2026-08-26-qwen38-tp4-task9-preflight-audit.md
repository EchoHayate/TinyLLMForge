# Qwen3.8-27B TP4 Task 9 preflight audit

## Outcome

Task 9 is complete as a source-bound, non-executing preflight:

```text
source: 116625a225d574c5561382df3d6e50f47eac27fa
branch: feat/kv-sparse-attention
attempt: 20260826-qwen38-tp4-communication-profile-r1
plan audit: PASS
strict-clean admission: READY
real dry-run: BLOCKED_KERBEROS_TTL
model checkpoint: MISSING
benchmark execution authorized: false
GPU worker started: false
```

This does not complete Task 10 or establish any correctness, throughput,
latency, communication-exposure, or overlap result.

## Root-cause repair encountered during preflight

The first real `--plan-only` invocation returned code zero from SSH but empty
stdout. The controller passed `sh`, `-c`, and the quoted command as separate
arguments after the SSH target. OpenSSH joins those arguments into one remote
command string, so the remote shell interpreted the command as:

```text
sh -c python3 -c <script>
```

The inner shell therefore executed only bare `python3`, which exited on EOF
without producing JSON. A RED regression test reproduced OpenSSH's joining
semantics. Commit `116625a225d574c5561382df3d6e50f47eac27fa` now passes one
fully quoted `sh -c ...` command argument after the SSH target.

Verification for the repair:

```text
focused regression: 1 passed
controller suite: 102 passed
Qwen3.8 related suite: 277 passed
py_compile: PASS
git diff --check: PASS
live ControlMaster sentinel: PASS
remote branch SHA: 116625a225d574c5561382df3d6e50f47eac27fa
```

## Preflight evidence

All runtime evidence is under:

```text
artifacts/qwen38_tp4_communication_profile/
  20260826-qwen38-tp4-communication-profile-r1/controller/
```

- `source_identity.json` proves local/remote source identity.
- `plan_only.json` is the complete immutable plan emitted by the controller.
- `plan_audit.json` inventories every absolute path and command argument.
- `ssh_storage_preflight.json` records the successful read-only SSH,
  hostname/user, mount state, space, fresh-tag state, and missing model.
- `strict_clean_admission.json` records the complete contemporaneous NVML
  inventory and selected GPUs.
- `dry_run.json` records the genuine Kerberos fail-fast outcome.
- `prompt_to_artifact_checklist.md` maps every Task 9 requirement to evidence.
- `manifest.sha256` binds all six JSON receipts and the checklist by content.

The selected rank map was:

```text
rank 0 -> GPU 0 -> GPU-57be086f-e967-c022-3832-93df4fc77bd0
rank 1 -> GPU 1 -> GPU-7dc22583-df04-6c76-4ba5-ea32c428c130
rank 2 -> GPU 2 -> GPU-63c05907-407b-8240-07a0-f38872840867
rank 3 -> GPU 3 -> GPU-f8904cb4-f9f0-c757-df36-e6fd971b3a9d
```

At admission time, every selected GPU recorded `0 MiB`, `0%`, and an empty
compute-process list. GPU 7 was not selected. No GPU was reserved, signalled,
or modified.

The approved mount had approximately 1.596 TB available. `findmnt` reported
both the older `/dev/nbd2 ro` layer and the later `/dev/nbd16 rw` layer.
No write probe was performed.

## Dry-run boundary

The real `--dry-run` returned:

```text
exit_code=2
classification=BLOCKED_KERBEROS_TTL
minimum_required_lifetime_seconds=5400
benchmark_execution_authorized=false
preserve_attempt=true
```

No `kinit` or `krenew` was run. The existing authenticated ControlMaster was
reused only for read-only probes. Its legacy `/tmp` socket is a local-only
variance caused by the unavailable Kerberos credential and the Unix socket
path-length failure encountered when trying a workspace-derived name; no task
logs or artifacts were written to local or remote `/tmp`.

## Remaining gates

Task 10 must not start until all of the following are true:

1. The expected Kerberos principal and TGT have at least 5,400 seconds left.
2. The real Qwen3.8-27B checkpoint exists under the approved remote root and
   every required file is verified against `model_manifest.json`.
3. Production correctness/workload/cleanup adapters are configured.
4. A fresh read of current NVML state still admits exactly four GPUs.

Communication/computation overlap remains unauthorized until the completed
producer and independent verifier both return exactly
`GO_COMMUNICATION_OVERLAP`.
