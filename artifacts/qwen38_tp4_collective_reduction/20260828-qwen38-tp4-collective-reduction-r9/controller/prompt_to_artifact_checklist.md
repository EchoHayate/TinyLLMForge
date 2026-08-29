# Qwen3.8-27B TP4 synchronous collective-reduction checklist

Date: 2026-08-28

This checklist belongs only to
`20260828-qwen38-tp4-collective-reduction-r9`. The attempt failed closed at
supervisor cleanup and has no terminal producer bundle. Worker-level PASS
entries below are not a gate success claim.

| Requirement | Evidence | Verdict |
| --- | --- | --- |
| Authoritative checkout | `/Users/bytedance/Desktop/TinyLLMForge` resolves to the active repository | `PASS` |
| Branch and remote | `feat/kv-sparse-attention`; `https://github.com/EchoHayate/TinyLLMForge.git` | `PASS` |
| Frozen source | `source_identity.json`: commit `6b7deaf5445879d7cf2626878f82a15626c19f77`, tree SHA-256 `bf3f61af819aeb2a0d1c41a11df9e9e46c279af53c20fd12ed919eaa7a314e7f` | `PASS` |
| Frozen model | remote `model_manifest.json`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `PASS_PREFLIGHT` |
| Approved storage | all attempt/source/case/bundle paths below `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818` | `PASS` |
| Canonical preflight lifecycle | original successful `dry_run.json` and `ssh_storage_preflight.json` restored; low-TTL retry retained separately as `resume_blocked.json`; regression commit `0ce14ff` | `PASS` |
| No root or remote `/tmp` task data | supervisor and all worker ranks have cwd below the r9 attempt source; task output is under the attempt directory | `PASS_OBSERVED` |
| Four strict-clean GPUs | physical GPUs 1-4, each 3 MiB, 0% utilization, and no compute process at launch | `PASS` |
| Exactly one worker launch | one supervisor PID and one worker process group for the frozen r9 identity | `PASS` |
| Runtime GPU ownership | target GPU PIDs observed as the worker plus three same-PGID rank children | `PASS_OBSERVED` |
| Calibration matrix | 3 workloads x 4 budgets x (2 warmups + 5 measured) = 84 cases | `PASS_WORKER` |
| Count-only terminal semantics | no passing budget maps to `selected_budget=null` and skips terminal workload cases | `PASS_CODE_AND_TESTS` |
| Selected event budget | `worker.json` records `selected_budget=null` | `PASS_WORKER` |
| Terminal workload matrix | correctly skipped because no nonzero event budget qualified | `PASS_WORKER` |
| Resource evidence | 8,141 snapshots, 84 samples, zero violations | `PASS_WORKER` |
| Worker result | `worker.json` classification `PASS`, return code zero | `PASS_WORKER` |
| Cleanup | process group destroyed and no owned children, but scans were `[[2837114, 2837115], [], []]` | `FAIL` |
| Supervisor result | fail-closed `supervisor_receipt.json` | `FAIL` |
| Root cause | substring scan matched transient probe source text; fixed for future attempts by `f4f6ee7` | `CONFIRMED` |
| Producer classification | assembler not run after supervisor failure | `NOT_PRODUCED` |
| Remote verifier | not run without a valid producer | `NOT_RUN` |
| Local verifier | not run without a valid producer | `NOT_RUN` |
| Immutable manifest | not produced for a failed attempt | `NOT_PRODUCED` |
| No async/overlap implementation | source prohibition scan and frozen plan flags | `PASS` |
| r9 authorization | no candidate-specific design may be authorized from this failed attempt | `DENIED` |
| Fix commit and push | exact-path staging, one required trailer, local/remote SHA equality at `f4f6ee7` | `PASS` |
