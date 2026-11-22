# LoRA recalibration v2 data preparation

Date: 2026-11-19. Status: **materialized and statically reviewed; structural acceptance blocked**.

## Purpose

Prepare twelve development tasks that expose functional mistakes and can later
support the paired feedback diagnostic. This is not a LoRA training run, a model
evaluation result, or a continuation of CED timing work. The tasks are not new
independent transfer/retention evaluation material.

## Material

- Dataset directory: `data/recalibration_v2/`.
- Twelve task definitions; 24 public and 72 private cases with manual derivations.
- Two separately authored reference decompositions per task.
- One initial incorrect program and two incorrect variants per task, with
  36 named, manually predicted rejection witnesses.
- 210 explicit structural comparisons: 96 against supervision, 48 against
  old public development tasks and 66 within this set.
- Candidate-visible prompts exclude private cases, derivations, references and
  incorrect variants. Stored file separation alone does not prove run isolation.

Independent case review confirmed all 96 expected values and closed one
derivation-only correction in C01/P01. Independent source review covered all
24 references and 36 incorrect programs, including manual checks of the 36
designated wrong outcomes. No reference, witness or public projection error was
found. These are static judgments, not executed passes.

The parent metadata audit passed all 60 ASTs, exact source/case bindings, public
projections, witness identities and 210 relation identities. The 154 existing
LoRA files retain their original aggregate hash. The original designs and v1
material are unchanged.

## Structural result

Independent review found two blocking near-template repairs:

1. **C01:S01:** changing only a time comparison's equality boundary solves the
   new initial program, just as in the supervision example. Reservation logic
   is already supplied correctly.
2. **C04:S03:** replacing begin-time `setdefault` with assignment of zero repairs
   all legal behavior. Interleaved transactions make the task description richer,
   but the required patch is still resetting an old accumulator.

C04's classification was corrected; neither initial program was silently
redesigned. Both blockers remain open. The full relation inventory is 2
near-template, 59 shared-mechanism and 149 different-structure judgments.
C10 retains a recency-mechanism limitation, without a new definite blocker
against the authorized comparison set.

The next proposed step is a narrow redesign of the C01/C04 initial defects,
keeping task semantics and correct cases unchanged where possible, followed by
new seals, hand-derived witnesses and renewed review. That redesign was not
performed here. In particular, swapping C01 to its existing rejected-occupancy
variant is only a possible revision, not a verified solution; any C04 change
must be compared again against the other reset tasks.

Review: `data/recalibration_v2/review.md`.
Snapshot bindings and explicit flags: `data/recalibration_v2/manifest.json`.
Detailed independent review and HTML finding report are in
`/private/tmp/lora-recalibration-v2-20261119/full-review/`.

## Boundaries

No task function, model, tokenizer, SSH/GPU or training was run. The 480 planned
fixed-source/case positions are future dynamic checks; none were executed here.
The separate proposed model diagnostic has 72 generations and at most 624
candidate executions; those are not observations from this preparation.

Dynamic correctness, actual tokenizer lengths, model difficulty, LoRA benefit,
formal evaluation readiness and runtime isolation are not established.
The two structural blockers prevent whole-set static acceptance. Checksum
sealing does not make this snapshot execution-ready, and the other ten tasks
have not been substituted as a smaller admitted experiment.

No old run is restarted; no branch, commit or push is made.
