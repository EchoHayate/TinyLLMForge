# Fixed Read-Only Route Collector

Date: 2026-11-20, Asia/Shanghai.

## Current Result

The revised private one-shot collector has passed 61 local tests with artificial
process, signal, selector and clock boundaries. The actual frozen local SSH
configuration/client identities and Include closure also matched in a
read-only file test. Independent first review rejected the 44-test version
with two P1 and one P2 findings. The second review accepted the publication
refinement but found adjacent helper-close cancellation, descendant cleanup
and claim-descriptor handoff defects. The third review accepted those fixes but
identified interrupted process construction losing an already-created child.
That failure was also reproduced and fixed. Independent fourth review approved
the exact current code/spec within the pinned-runtime contract, with no
remaining supported P0/P1/P2 findings. This accepts the local implementation,
not a real observation or permission to execute one.

No SSH connection, credential-content read, remote command, stored task,
tokenizer/model load, GPU operation or training was performed. The real
one-shot claim does not exist. No historical component suite was rerun.

This is collection infrastructure, not a new LoRA result.

## Implemented Boundary

- Accept only the exact reviewed r2 scope bytes. No arbitrary command, retry,
  transport callback, CLI or authorization boolean is exposed.
- Check six pinned files plus ordered Include edges, glob membership,
  directory membership and resolved targets before preparation, immediately
  before dispatch and after collection.
- Consume one exclusive scope-specific claim in an owned mode0700 private
  parent. Use exclusive no-follow mode0600 files and file/directory fsync.
  Preparation failures do not launch a process; the claim is never retried.
- Replace the environment and use the frozen argv, DEVNULL stdin, cwd `/`,
  closed unrelated descriptors and a new owned process group.
- Retain at most 16,384 stdout and 131,072 stderr bytes. Record overflow
  separately instead of accepting a truncated prefix.
- Budget 20 seconds from before spawn and a separate shared 2-second cleanup
  window. EOF, exit, direct-child reaping and local-group absence are separate.
  Cancellation is re-raised after attempted cleanup and evidence persistence.
- Keep parsing, configuration identity, process outcome and scientific
  admission distinct. No observation grants execution permission or proves
  remote isolation, absence of remote descendants, tokenizer fit or LoRA gain.

These are cooperative local-filesystem pre/post observations, not an atomic
snapshot or protection against a malicious same-UID actor. Observed overruns
cannot qualify success, but blocking operating-system calls are not subject
to a hard-real-time guarantee. Artificial process tests are not evidence of
actual SSH/process-group behavior on the target route.

## Local Evidence

Private root: `/private/tmp/lora-route-collector-20261120-QF2RTX`.
Internal routes, argv and raw diagnostics are not reproduced here.

| Evidence | Observed result |
| --- | --- |
| `red.log`, `red.exit` | 44 expected absent-implementation assertion failures, exit1 |
| `green-attempt-1.log`, `green-attempt-1.exit` | 44 passed, no errors or skips, exit0 |
| `implementation-verification.json` | 230 prior files byte-identical before document integration |
| `fresh-review.md` | Original rejection: 2 P1, 1 P2 |
| `review-p1-reap-{red,green}.log/.exit` | Early-abandonment reproduced, then focused pass |
| `review-p1-finalization-{red,green}.log/.exit` | Lost primary cancellation/missing artifacts reproduced, then 3 passes |
| `review-p2-publication-{red,green}.log/.exit` | Success-shaped partial-finalization artifacts reproduced, then 3 passes |
| `green-revised.log`, `green-revised.exit` | 51 passed, no errors or skips, exit0 |
| `revision-verification.json` | 229 old files unchanged, old ledger byte-prefix preserved |
| `fresh-review-r2.md` | Second rejection: helper cancellation, descendant cleanup, descriptor handoff |
| `review-r2-group-{red,green}.log/.exit` | Reaped-child/unknown-group early exit reproduced, then focused pass |
| `review-r2-files-{red,green}.log/.exit` | Three helper-cancellation and one descriptor-handoff paths reproduced, then 4 passes |
| `green-r3.log`, `green-r3.exit` | 57 passed, no errors or skips, exit0 |
| `fresh-review-r3.md` | Prior fixes accepted; constructor ownership gap rejected |
| `review-r3-spawn-{red,green}.log/.exit` | Three constructor ownership/cancellation paths reproduced, then 3 passes |
| `green-r4.log`, `green-r4.exit` | 61 passed, no errors or skips, exit0 |
| `fresh-review-r4.md` | Approved exact source/test identities under the documented limits |
| Public interfaces | `validate_scope`, `parse_observation`, `collect_once` |
| Production imports | Standard library only |
| Real attempt claim | Absent |
| Independent re-review | Approved; not runtime or research acceptance |

The append-only ledger will change during integration; the original 230-file
manifest and its earlier verification remain immutable historical snapshots.
Final integration must verify the old ledger as a byte prefix, not replace its
old hash with a new one.

Accepted source SHA256:

- `route_observation_collector.py`:
  `175515df2d32afbf7c6e13ef0c48e58223ce60d663e163b6dbdbe9c31af71a4b`
- `test_route_observation_collector.py`:
  `5d0e7d30f0302e4dd1ddc9fa0a2ee973dab3174108b6f853ac3018d00564f210`

The independent reviewer read the actual source, tests and installed dependency
as text and inspected the saved logs; it did not rerun tests or operate a real
process. Parent final sealing verifies unchanged approved code and prior-file
preservation without replaying the same suite.

## Receipt Consumer Rule

A file cannot attest its own final successful fsync. `receipt.json` is therefore
always marked `UNFINALIZED_CANDIDATE`, with `observation_complete=false` and
`persistence_completed=false`. It records the proposed observation outcome in
`candidate_observation_complete`; that field is not accepted completion.

Only a normal collector return reports completed persistence, includes the
exact candidate-receipt SHA256, and may report observation completion. A later
consumer needs a separately captured successful caller return and matching
artifacts, not merely an existing or parseable receipt. No such caller capture
or real observation has been produced by this task.

Receipt/close failures attempt a separate exclusive phase/type failure note.
This does not promise that failing storage can persist its own failure, and
does not rewrite an earlier artifact into a more favorable result.

## Constructor Boundary

The collector retains its process object before calling its constructor once.
A constructor error after a recorded child PID no longer loses the handle
needed for cleanup. If session creation may still be pending, it may signal
only that retained direct child as well as the owned group, with separately
recorded attempts. A pre-PID exception retains unknown ownership rather than
being reported as proof that no child was started.

This ownership mechanism is checked against the inspected local Python3.9.6
and fixed standard-library subprocess source. Different runtime/source
identities reject before dispatch; this is not a portable launcher abstraction.
The exact remote scope/argv is unchanged. Arbitrary instruction-level
asynchronous-exception atomicity and recovery from process termination are
outside the guarantee.

## Research Status

The existing November16 result records a real 16-step Qwen3-8B LoRA
training/save/reload/unload qualification. This engineering step is already
complete and must not be represented as unperformed or rerun for this collector.

The November17 development result records base and text each passing 4/4
tasks. It contains no LoRA comparison; its `INCONCLUSIVE_CALIBRATION` result
remains unchanged. Those historical results were checked against the retained
repository reports, not reconstructed here from the now-unavailable original
private run directory.

The relevant next evidence is real execution, not more instances of the same
synthetic tests:

1. Reviewed collector plus separately granted exact-scope permission for the
   single read-only route observation. Permission is still missing.
2. Actual observation, then separately scoped runtime/isolation preflight and
   fixed-source supervisor/controller with complete launch binding.
3. Accepted new 480-item dynamic evidence, followed by real 120-row tokenizer
   measurements and the frozen base/text feedback/retry diagnostic.
4. A separately frozen same-information `lora_text` versus `text` comparison
   with new transfer/retention tasks and predeclared quality/cost criteria.

No component result substitutes for these original observations. LoRA need
not win; a reliable zero or negative result is acceptable. Neither a benefit
nor a lack of benefit has been established by this collector.
