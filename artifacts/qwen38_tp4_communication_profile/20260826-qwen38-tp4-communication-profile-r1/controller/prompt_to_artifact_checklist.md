# Qwen3.8-27B TP4 Task 9 prompt-to-artifact checklist

Date: 2026-08-26

Terminal classification:

```text
TASK9_PREFLIGHT=COMPLETE
TASK10_LAUNCH=BLOCKED_KERBEROS_TTL_AND_MODEL_MISSING
BENCHMARK_EXECUTION_AUTHORIZED=false
GPU_WORKER_STARTED=false
COMMUNICATION_OVERLAP_AUTHORIZED=false
```

| Requirement | Evidence | Verdict |
| --- | --- | --- |
| Use only the authoritative checkout and branch | `source_identity.json` records `feat/kv-sparse-attention` | PASS |
| Bind execution to committed source | `source_identity.json` and `plan_only.json` bind `116625a225d574c5561382df3d6e50f47eac27fa` | PASS |
| Push implementation and match GitHub branch | `source_identity.json` records identical local and remote SHA | PASS |
| Preserve exactly one required commit trailer | `git log -1 --format='%H%n%B'` inspected for `116625a`; one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer | PASS |
| Reuse or rebuild SSH without `kinit` or `krenew` | `ssh_storage_preflight.json` records a successful read-only probe through `/tmp/ssh-sitian-10.232.195.203`; `kerberos_initialization_performed=false` in `dry_run.json` | PASS WITH DOCUMENTED LOCAL SOCKET VARIANCE |
| Record remote hostname and user | `ssh_storage_preflight.json`: `n232-195-203`, `sitian` | PASS |
| Record `/data00/home/sitian` mount state | `ssh_storage_preflight.json` records `/dev/nbd2 ro` and `/dev/nbd16 rw` mount rows | PASS |
| Do not write task data outside the approved remote root | `plan_audit.json` checks every absolute plan path; `path_violations=[]` | PASS |
| Do not write task data to remote `/`, `/tmp`, `/private/tmp`, the old checkout, or adaptive-ngram | `plan_audit.json` checks all 11 absolute plan paths; all are beneath the approved root | PASS |
| Use a fresh attempt tag | `ssh_storage_preflight.json` records `attempt_exists=false` for `20260826-qwen38-tp4-communication-profile-r1` | PASS |
| Inspect every planned argv element | `plan_audit.json` records all four commands and `command_violations=[]` | PASS |
| Do not emit Kerberos initialization or process-signalling commands | `plan_audit.json` checks `kinit`, `krenew`, `kill`, `pkill`, and `killall`; none appear | PASS |
| Do not start a GPU worker during preflight | `plan_audit.json` records `worker_command_count=0`; `dry_run.json` records `gpu_worker_started=false` | PASS |
| Keep benchmark execution unauthorized | `plan_only.json`, `plan_audit.json`, and `dry_run.json` all record `benchmark_execution_authorized=false` | PASS |
| Select exactly four GPUs from current state | `strict_clean_admission.json` records the full inventory and selected GPU indices `0,1,2,3` | PASS |
| Enforce memory `<=1024 MiB` | Each selected row in `strict_clean_admission.json` records `0 MiB` | PASS |
| Enforce utilization `<=5%` | Each selected row in `strict_clean_admission.json` records `0%` | PASS |
| Enforce no compute processes | Each selected row in `strict_clean_admission.json` records `compute_processes=[]` | PASS |
| Preserve external GPU work | The full inventory in `strict_clean_admission.json` excludes occupied GPU 7; no signal/reservation command was emitted | PASS |
| Bind the declared model revision | `plan_only.json` records `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | DECLARED, NOT YET ACQUIRED |
| Verify a real immutable model checkpoint | `ssh_storage_preflight.json` records `model_root_exists=false` | BLOCKED FOR TASK 10 |
| Run a genuine controller dry-run | `dry_run.json` records exit code `2` and `BLOCKED_KERBEROS_TTL` | PASS AS FAIL-FAST EVIDENCE; NOT READY |
| Avoid hidden fallback when Kerberos is unavailable | `dry_run.json` records `preserve_attempt=true`, no remote write, and no worker start | PASS |
| Produce performance benefit and cost | No workload was authorized or run | NOT APPLICABLE TO TASK 9; REQUIRED IN TASKS 11-12 |
| Authorize communication/computation overlap | No producer/verifier profiling verdict exists | NOT AUTHORIZED |
| Bind the Task 9 receipts by content hash | `manifest.sha256` covers all six JSON receipts and this checklist | PASS |

The current blockers are external prerequisites, not successful model or
performance evidence:

1. The local Kerberos payload is only `{ "version": 1 }`; the required
   `sitian@BYTEDANCE.COM` principal and a TGT with at least 5,400 seconds
   remaining are absent.
2. The declared Qwen3.8-27B snapshot directory does not exist below the
   approved remote root.
3. Production worker adapters are not configured, so a non-dry campaign
   remains unavailable even after the first two prerequisites are restored.

No Task 10 correctness claim, Task 11 performance claim, or overlap design
is authorized by these receipts.
