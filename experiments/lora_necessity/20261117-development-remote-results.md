# Fixed base/text development run

Date: 2026-11-17. Run: `46be394c23df43878a50f5f5676d1648`.

**The remote run completed and offline reconstruction verified its evidence.
Development calibration did not pass: both arms answered all four tasks
correctly. Independent post-run evidence review passed; the result document was
approved without P0–P2 findings.**

This is a completed development experiment, not evidence for or against LoRA.
The model was Qwen3-8B, loaded remotely on the approved A100 80GB GPU. No LoRA
was loaded or trained, and no serving-engine integration was exercised.

## What ran

The independently admitted package executed once, from 17:04:06 to 17:09:31
Asia/Shanghai. The transport recorded one invocation, exit zero, no timeout,
no failure and 325.375 seconds elapsed. The complete export contains 958 files.
The claim, raw streams and original receipts remain unchanged.

The fixed order was D01 base/text, D02 text/base, D03 base/text, D04 text/base.
Each episode had two generation calls, even when its first candidate passed
the public checks. Base received the public task; text additionally received
the original frozen supervision examples. Only public execution feedback was
available for the second round. The separate final scorer used the frozen
eight-case inventory for each episode.

| Observed quantity | Result |
| --- | ---: |
| Completed episodes | 8 / 8 |
| Completed generation calls | 16 / 16 |
| Verified public execution receipts | 16 |
| Verified final execution receipts | 64 / 64 |
| Passed final cases | 64 / 64 |
| Missing or invalid final cases | 0 |
| Unknown public/final child starts | 0 |
| Observed completed forward calls | 508 |

The 64 final cases comprise four tasks evaluated by both arms, not 64 independent
tasks. Public and final execution counts are separate; the final inventory also
includes public cases and should not be presented as 64 unseen cases.

| Arm | Final tasks passed | Final cases passed |
| --- | ---: | ---: |
| Base | 4 / 4 | 32 / 32 |
| Text memory | 4 / 4 | 32 / 32 |

All sixteen first-round public checks also passed. The recorded feedback exposed
no first-round error to correct, so this run does not demonstrate recovery from
an observed mistake. Those public checks alone do not establish first-round
correctness on the separate scoring cases, which were not run in that round.

## Cost, not a performance benchmark

These are sums of recorded calls in one fixed run. They are not repeated
throughput measurements or independent randomized performance estimates.

| Recorded quantity | Base | Text memory |
| --- | ---: | ---: |
| Generation calls | 8 | 8 |
| Input tokens across both rounds | 1,974 | 17,656 |
| Output tokens | 260 | 248 |
| Generation wall seconds | 33.245 | 124.819 |
| Observer elapsed seconds | 33.285 | 124.856 |
| Public plus final execution seconds | 9.542 | 8.778 |

The text arm used more input tokens and more measured generation time without
a higher final score on these saturated tasks. This does not establish a general
cost disadvantage, the absence of text-memory benefits, or a LoRA advantage.

The shared coordinator wall time was 201.412 seconds. Shared preparation was
85.237 seconds; final integrity checking was 31.865 seconds. The transport wall
time was 325.375 seconds. Nested timers overlap: do not add these fields or
assign all shared preparation to either arm. The original driver fields
`driver_wall_ns` and `terminal_emit_ns` remain null; transport and exported wall
observations are separately named measurements, not replacements for them.

Initial and final base/tokenizer identities matched the approved identities.
Final RNG preservation was verified and no integrity violations were reported.
The unchanged base digest supports the no-weight-update boundary of this run.

## Calibration and decision

The accepted verifier returned:

- `process_status=completed`
- `verification_status=verified`
- `calibration.status=INCONCLUSIVE_CALIBRATION`
- sole calibration reason: `both_arms_all_pass`
- no protocol contradictions or incomplete-evidence reasons

This is the predeclared ceiling condition in
`docs/superpowers/specs/2026-11-17-lora-development-coordinator-design.md`
and the original pilot design, not a new threshold selected after the results.
The observed result is consistent with development tasks that do not distinguish
the two arms; it does not prove the entire task distribution is easy.

The current batch stops here. Do not train or compare LoRA, run the 32-task
evaluation, change prompts, replay this run, or relax the calibration rule to
turn it into a pass. A future study needs a separately frozen development
calibration design with discriminatory tasks and explicit cost/retention
criteria. Transfer or retention outcomes must not be opened to select those
tasks. Any revised study must retain this saturated result as prior development
evidence rather than call reused tasks independent evaluation.

`benchmark_eligible`, `formal_evaluation_ready`, `net_benefit_established` and
`isolated` all remain false. The last flag is the conservative study-level
readiness flag; reviewed per-child execution checks are not a claim of general
hostile-code safety.

## Evidence

Private root: `/tmp/tlf-candidate-executor-20261117-gb1gZX`.
Internal SSH routes and credential information are not reproduced here.

| Artifact under private root | SHA256 |
| --- | --- |
| `remote-development-v1/package.json` | d926d802e724d56930bd8f9296820ac5cf20715604e8f308c1fd439c411c02c0 |
| `remote-development-v1/preparation.json` | 1d3b8a585ba911348fd062654f9a057d97403573a4433bdae10472fa21b24b14 |
| `remote-development-v1/transport/transport-claim.json` | 8c639c980d1c4d58ab75830ce59a112bc5a025bc7438a76181d4582a94951804 |
| `remote-development-v1/transport/transport-receipt.json` | 42bed9769ca7e9f9d877f5d275889e2510acd64abbbee429f2bef020da27c4dd |
| `remote-development-v1/transport/stdout.bin` | 257585ffeb33f5adac1e5284847661d7c3d4183668e25171fadb3d5dd0d350a8 |
| `remote-development-v1/transport/stderr.bin` | 756fb88e3dd7b0fe3ddec58b16b26efdae216137b55513f2e2970b81a3bc7369 |
| `remote-development-v1/verification/development-remote-verification.json` | 527c06f3307f4b53b8a69c785523093efca6d83fa6dfda6f42faf6f7cfb321e3 |

The raw streams contain 8,309,070 stdout bytes and 58,666 stderr bytes.
`development-remote-launch.log` retains the outer command's status and timestamps.
`development-remote-reconstruction.log` retains source/input/stream checks and
the accepted offline verifier output. All 44 preparation input/artifact pins
were rechecked before reconstruction. Only pinned pure helpers were interpreted
locally; candidate and reference programs were not executed locally.

The earlier 439-test synthetic component regression is separate evidence,
described in `20261117-development-remote-package-results.md`; it is not counted
as additional real-model observations.

Independent review: `development-remote-result-review.md` under the private
root, SHA256
`fd07990ed3e5ba656a93c01412e9c0e2bd5ade35040ba045f616251a6100195c`.
Its `development-remote-result-review/final_comments.json` is empty.
The review independently reconstructed the export in memory with stdlib
JSON/base64/hash/AST checks, without running the product verifier, model,
candidate/reference code or remote commands. It checked all 80 candidate-bound
receipts, original/second-round messages, calibration and costs.

The reviewed document snapshot is retained as
`development-remote-result-review/reviewed-result-document.md`, SHA256
`ed7f76976e98e2042f540de12ebf989abf91d2e49a4efc2851d7319b2e8d7fb6`.
Only the review-status wording and this evidence note were added afterwards.
Main accepts the process/evidence result, not calibration success or permission
for another experimental run.
