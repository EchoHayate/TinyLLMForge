# Qwen3.8-27B TP4 synchronous collective-reduction checklist

Date: 2026-08-29

This checklist belongs only to
`20260828-qwen38-tp4-collective-reduction-r10`. The attempt uses the
post-r9 argv-aware cleanup scanner and completed on 2026-08-29. The
classification is a qualification result, not a measured speedup claim.

| Requirement | Evidence | Verdict |
| --- | --- | --- |
| Authoritative checkout | `/Users/bytedance/Desktop/TinyLLMForge` resolves to the active repository | `PASS` |
| Branch and remote | `feat/kv-sparse-attention`; `https://github.com/EchoHayate/TinyLLMForge.git` | `PASS` |
| Frozen source | `source_identity.json`: commit `f4f6ee7a9182f47d5e4f6577c217db1aa9793391`, tree SHA-256 `b478a9b3c59d11f7b1fb94f2d2530110d40d3f44cc77b4341b6a5c85eed4f83a` | `PASS` |
| Frozen model | remote snapshot revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, config/index plus 18 weight shards readable | `PASS_PREFLIGHT` |
| Approved storage | all attempt/source/case/bundle paths below `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818` | `PASS` |
| Kerberos admission | TGT expiry `2026-08-29T16:33:34+08:00`; 27,832 seconds remaining at launch guard | `PASS` |
| Four strict-clean GPUs | physical GPUs 1-4, each 3 MiB, 0% utilization, and no compute process at launch | `PASS` |
| Exactly one worker launch | one supervisor PID `2850309`, worker PID/PGID `2850310`, and no replacement worker | `PASS` |
| Runtime GPU ownership | 8,106 supervisor snapshots, 84 resource samples, and zero violations | `PASS` |
| Calibration matrix | 3 workloads x 4 budgets x (2 warmups + 5 measured) = 84 cases | `PASS_84_OF_84` |
| Selected event budget | every nonzero budget violates at least one frozen overhead ceiling; selected budget is `null` | `PASS_NO_SELECTION` |
| Terminal workload matrix | correctly skipped because no nonzero event budget qualified | `PASS_CONDITIONAL_SKIP` |
| 130-site inventory | 130 static catalog rows and 130 consumer-dependency proofs | `PASS_130_OF_130` |
| Exact correctness | 560 calibration correctness rows; verifier reports valid exact correctness | `PASS` |
| Four-rank census agreement | no terminal census rows were authorized; verifier accepts the exact conditional zero-row inventory | `PASS_CONDITIONAL_SKIP` |
| Benefit and cost | no measured latency benefit; one static removable embedding all-reduce costs 1,907,097,600 persistent and peak device bytes per rank | `INCONCLUSIVE_COST_DOMINATED` |
| Cleanup | process group destroyed, no owned children, and three empty argv-aware exact-tag scans | `PASS` |
| Producer classification | `INCONCLUSIVE_PROFILER_OVERHEAD` | `PASS_RECORDED` |
| Remote verifier | status `PASS`, reconstructed classification matches producer | `PASS` |
| Local verifier | SHA-256 `c115dee2...`; byte-identical to remote verifier | `PASS` |
| Immutable manifest | 16 hashed artifacts plus `manifest.sha256`; local verifier rehashed every artifact | `PASS` |
| No async/overlap implementation | source prohibition scan and frozen plan flags | `PASS` |
| Controller timeout recovery | initial 120-second assembler call timed out after the remote producer continued to completion; same-attempt POSTPROCESS resume completed without a second worker | `PASS_RECOVERED` |
| Post-gate timeout hardening | commit `d8d85da9479a06f122a93987b1d87b9a5f8e0cd0` gives postprocess commands at least 600 seconds | `PASS` |
| Final audit and handoff | independent audit, Phase 1 reconciliation, and one literal terminal handoff block | `PASS` |
| Commit and push | exact-path staging, required single trailer, and local/remote SHA verification | `PASS_AT_HANDOFF` |
