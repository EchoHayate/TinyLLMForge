# LoRA recalibration v2 r1 — static revision review

Date: 2026-11-19. Status: **actual-source static review passed; dynamic and
tokenizer validation pending; no experiment execution authorized**.

## Scope and result

This revision implements the user-confirmed C01 and R3 C04 initial defects.
It does not change the task specifications, correct answers, case inputs,
reference solutions, preserved variants, or ten other initial programs.
The original `recalibration_v2/` directory is preserved as a blocked historical
snapshot; its F01/F02 findings are not rewritten or closed in place.

Only two of the 60 embedded source functions change:

- C01 makes its capacity decision before clearing expired reservations.
  Expiration comparison, exact-capacity comparison and accepted-only accounting
  are correct; the intended fault is reading stale occupancy.
- C04 uses one global total minus each transaction's begin snapshot, and rolls
  that global total back on rollback. Target-ID lookup and output order are
  correct; the intended fault is mixing independently owned amounts.

These are intentionally incorrect initial programs for a development diagnostic,
not corrected reference implementations. The independent reviewer found they
faithfully implement the approved faults, without new accidental defects.

## Actual-source review

Franklin, agent `01a1f26c-ae6f-7e83-82bf-9ce2044047b7`, read both actual new
initials and manually derived their eight case results each, including the two
P01 rejection witnesses and behavior differences from the preserved variants.
The reviewer rechecked all 45 affected actual-source repair relationships:
22 involving C01 but not C04, and 23 involving C04 including C01:C04 once.
Spec compliance and static quality both passed; confirmed findings: zero.

The 45 newly reviewed relationships are 0 near-template, 18 shared-mechanism
and 27 different-structure. The other 165 records remain exact copies of the
original reviewed data: 44 shared-mechanism and 121 different-structure.
The complete table therefore contains **0 / 62 / 148** across 210 pairs.
This is not a claim that all 210 pairs were newly reviewed, nor proof of
statistical independence, pretraining novelty or model difficulty.

Small correct repairs are explicitly allowed. C01 can move the filter or use
current-time validity in its sum. C04 can compensate other active snapshots
for each global-total change, keep independent amounts, or reconstruct a
lifetime from the complete input. No minimum patch length or mandated data
structure is imposed. C10's original reconstruction caveat remains unchanged.

Independent report:
`/private/tmp/lora-recalibration-v2-r1-20261119/independent-review.md`,
SHA256 `8d0b00a600b3683adabadde5450eb04130e816416ef3b5a337d8612ba5737bf7`.
Its structure-file snapshot is the pre-finalization file, SHA256
`bf5dfaacb534c7870e549f7c922b170a7ae220909941f40aca55ef22623974b3`.
Only review-status, reviewer/hash and explicit author metadata changed after
that review. The stable reviewed content (tasks and all three relationship
collections) retains SHA256
`96c6b714ff9410fd9b27f96acf48bba32283a91f8619a3a662ce809f4f9b710c`.
Neither source file changed after independent review.

## Authorship and inherited evidence

New initials: Feynman, agent `01a1f335-74a8-7de2-8f34-e9490707cd47`,
transcribing the user-confirmed design.

| Source | SHA256 |
| --- | --- |
| C01 initial | `a3e35150a15d563c7b9443c4665a3b4428856d4ca150199b125c7b6921da7416` |
| C04 initial | `70f46359eab48970c718982d689d191f6c443b17bdd8e092ae6b6f05716f7a60` |

Original reference A, variants and ten unchanged initials retain Kant's
authorship; reference B retains Huygens's authorship. The original A/B seals
and before-integration independence records apply to those inherited sources,
not to these two new initials. No new reference authoring or independent
human sign-off is claimed.

The 96 correct cases and the other 58 sources retain the original static
review evidence, backed by exact preservation. Their complete semantic review
was not repeated here. The two newly reviewed eight-case trajectories are
incorrect-initial predictions, not 16 new correct oracle cases.

Implementation report:
`/private/tmp/lora-recalibration-v2-r1-20261119/implementation-report.md`,
SHA256 `937500feb6b96ebc76764b78270df2ebc0959abac3e18ae834997e8926183a12`.
It records the 16 hand predictions and inspection history, including a corrected
inspection-script invocation error. That error did not require a data change.

## Preservation and metadata checks

Before creating r1, an explicit list froze 169 existing files: all 160 existing
LoRA files plus nine sealed specifications, reviews and the original plan.
Every listed path retains its SHA256. The original 154-file subset also
retains its historical aggregate
`e9b4510aa7247fe926764d6ff43a4d1b046fd09fa8ad19c5680243ee1137792e`.

Explicit list:
`/private/tmp/lora-recalibration-v2-r1-20261119/baseline-files.json`,
SHA256 `db11cf806307e83c149c88d507f8076166cd8cccd4125e2e01d2a8477708ccfc`.
The 160-file LoRA aggregate is
`c4103a653cc018f47423f56cd79483787bc15584eb7304596250cf449b26a7f7`.
Checks use this explicit list rather than rediscovering all paths outside r1.

Parent metadata inspection verified all 60 ASTs/signatures, exactly two changed
source identities, 58 unchanged complete source records, 96 unchanged complete
case objects, 24 public projections, 36 witness bindings, all 210 relation
identities and 165 unchanged relation records. Comparisons preserve JSON types,
so booleans cannot silently equal integers. Public prompts retain their exact
schema and expose no private cases, references or variant metadata.

Inspector:
`/private/tmp/lora-recalibration-v2-r1-20261119/audit_revision.py`.
It reads stored functions only as strings/ASTs; it never invokes them.
Final manifest/hash/status verification uses the same inspector with `--final`.

## Historical findings and remaining gates

| Item | Disposition |
| --- | --- |
| Original F01, C01:S01 | Still recorded in unchanged v2; revised C01 has a different actual repair, reviewed as shared mechanism. |
| Original F02, C04:S03 | Still recorded in unchanged v2; revised C04 requires ownership isolation, reviewed as shared mechanism. |
| Rejected R1/R2 C04 designs | Retained with their rejection reviews; neither algorithm was implemented in r1. |
| New static findings | None confirmed in the bounded actual-source review. |
| Dynamic correctness | Pending; zero of the 480 planned source/case positions executed. |
| Tokenizer lengths | Pending; no tokenizer loaded. |
| Runtime isolation, model difficulty, LoRA benefit | Not established. |

All task-source, model-generation, tokenizer-load, training-step and remote-run
counts remain zero. All execution authorization, benchmark eligibility, formal
evaluation readiness, net-benefit and isolation flags remain false. File
separation is not runtime isolation; manual traces are not execution receipts.
This dataset is development material, not independent LoRA transfer/retention
evaluation or a training corpus. No runtime adapter, model driver or training
configuration was changed.
