# LoRA Initial-Defect Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Materialize the user-confirmed C01 and R3 C04 initial defects in a separate static dataset, preserving all correct material and the original blocked snapshot.

**Architecture:** Create `experiments/lora_necessity/data/recalibration_v2_r1/` with the same five-file layout as v2. Only two embedded initial functions and their defect/witness records change in prompt/oracle data. Replace the 45 affected structural judgments, retaining the other 165 and their original review provenance.

**Tech Stack:** UTF-8 JSON/Markdown and Python standard-library JSON, AST, tokenize and SHA256 inspection; no task execution.

## Global Constraints

- The user's “继续” follows the presentation of C01 and R3 C04 and confirms implementation of these specific designs, not model execution.
- Work in the current `experiments/feedback-active-verification` checkout; no new branch, commit, push, installation or cleanup of existing untracked files.
- Do not execute/import/compile-to-run embedded task source, load a model/tokenizer, use SSH/GPU or resume an old session. AST parsing is permitted, not dynamic validation.
- No application runtime or unit-test suite is being added in this slice. Deliberately wrong programs are data; do not “repair” their specified defects.
- Preserve original v2 five files, old results, approved designs, R1/R2/R3 designs and reviews byte-for-byte. Historical F01/F02 and rejected R1/R2 remain unchanged.
- Preserve all 96 complete case objects, all 12 specifications/entry points/public projections, 24 reference records, 24 variant records and ten other initial records. Exactly two of 60 source functions change; the other 58 remain byte-identical.
- No inline source comments or single-letter variable names.
- Structural comparisons use S01–S08 and public D01–D04 only; do not read old transfer/retention answers or model results.
- Static, dynamic and tokenizer-length states remain separate. All execution counts are zero and run/readiness/benefit/isolation flags remain false.
- Use apply_patch for edits. Capture an explicit baseline file list before creating the revised dataset; subsequent preservation checks use that list, not an exclusion-based rediscovery.

## Inputs

- C01: `docs/superpowers/specs/2026-11-19-lora-recalibration-initial-defects-design.md`, section 3; SHA256 `68bf2f146d8d0d638df18062425ae0c7896b4e214cdf442d8675cf1ff0631c39`.
- C04: `docs/superpowers/specs/2026-11-19-lora-recalibration-initial-defects-design-r3.md`; SHA256 `c6950e076df69eb33cd200f188f5c71a3e286206b3abd3b9fb80bcca90899102`.
- R3 review: `docs/superpowers/specs/2026-11-19-lora-recalibration-initial-defects-review-r3.md`; SHA256 `ad3f4857002b6a4ea5f9e85b17a94c7db85ff5c00be9268d200618276bd4ac93`.
- Original manifest: `experiments/lora_necessity/data/recalibration_v2/manifest.json`; SHA256 `1e3e9564e3d0862e2263961289246545deb87a18d080ba67622c0bd7dadb8dfb`.

## Task 1: Preservation and revised data

**Create:**
- `/private/tmp/lora-recalibration-v2-r1-20261119/baseline-files.json`: explicit repository-relative paths and SHA256 digests for the preexisting LoRA files and sealed input designs/reviews.
- `experiments/lora_necessity/data/recalibration_v2_r1/prompts.json`
- `experiments/lora_necessity/data/recalibration_v2_r1/oracles.json`
- `experiments/lora_necessity/data/recalibration_v2_r1/structure.json`
- `/private/tmp/lora-recalibration-v2-r1-20261119/implementation-report.md`

**Interfaces:** Preserve the original prompt/oracle schemas exactly. Structure retains all 210 relationships, keyed by task IDs in the original order. Add revision/provenance metadata only in structure or manifest, not in public prompts.

- [x] Capture the explicit baseline list before creating the revised directory; verify old v2 bindings and the 154-file old aggregate.
- [x] Inspect the target's absence as the pre-materialization check. This is not a behavioral RED test; dynamic execution remains prohibited.
- [x] Materialize copies with only the following two source replacements:

```python
def admit_reservations(capacity, requests):
    active = []
    decisions = []
    for time, size, expires in requests:
        occupied = sum(reservation[0] for reservation in active)
        accepted = occupied + size <= capacity
        decisions.append(accepted)
        if accepted:
            active.append([size, expires])
        active = [reservation for reservation in active if reservation[1] > time]
    return decisions
```

```python
def transaction_totals(events):
    total = 0
    starts = {}
    committed = []
    for operation, transaction, amount in events:
        if operation == 'begin':
            starts[transaction] = total
        elif operation == 'add':
            total += amount
        elif operation == 'commit':
            committed.append([transaction, total - starts.pop(transaction)])
        elif operation == 'rollback':
            total = starts.pop(transaction)
    return committed
```

- [x] Change only those initials' defect descriptions and P01 witness descriptions/predictions. C01 predicts `[true,false,false,true,true]`; C04 predicts `[["a",0],["b",0]]`. Both remain hand-derived, not execution receipts.
- [x] Replace C01's 22 relationships not involving C04 from the R1 table, and all 23 C04 relationships from the R3 table. Store specific minimum-repair reasons; do not copy rejected C04 wording from R1. Preserve all other 165 records exactly.
- [x] Update only C01/C04 repair obligations and obsolete concerns. Preserve C10's existing caveat. Record new initial authorship separately from inherited A/B/variant authorship.
- [x] Mark new actual-source review pending, not approved from design alone. Record 16 hand predictions in private review evidence, not public prompts.
- [x] Inspect JSON, all 60 ASTs and signatures, 24 public projections, 36 witness bindings, 58 unchanged source identities and 96 complete cases. Compare typed JSON values so booleans cannot silently equal integers.
- [x] Report exact new file digests and validation output; no commit.

## Task 2: Independent actual-source review

**Create:** `/private/tmp/lora-recalibration-v2-r1-20261119/independent-review.md`.

**Consumes:** Task 1 artifacts, this plan, original v2, approved C01/R3 design and original review provenance. Base and head are the same unchanged commit; review untracked old/new data differences, not an empty Git diff.

- [x] Reviewer checks actual two source strings against the approved algorithms and manually derives all 16 outputs.
- [x] Recheck two public rejection witnesses and differences from retained variants. Inspect the 45 actual-source relationships, permitting small valid repairs including C04's unified snapshot compensation.
- [x] Distinguish newly reviewed 45 pairs/two initials from inherited 165 pairs/58 unchanged sources; check preservation evidence.
- [x] Return separate spec-compliance and static-quality verdicts. Resolve actual findings before marking static review complete; preserve findings rather than lowering gates.

## Task 3: Static seal and closeout

**Create:**
- `experiments/lora_necessity/data/recalibration_v2_r1/review.md`
- `experiments/lora_necessity/data/recalibration_v2_r1/manifest.json`
- `experiments/lora_necessity/20261119-recalibration-initial-defects-results.md`

**Modify:** `.superpowers/sdd/lora-recalibration-initial-defects-20261119.md` and plan checkboxes only.

- [x] Write scope-limited review status based on Task 2, with inherited evidence and historical F01/F02 kept distinct.
- [x] Manifest binds the other four files, original manifest, sealed designs/reviews, explicit preservation list, new initial hashes/authorship and independent actual-source review.
- [x] Recompute all 210 labels from actual records; report zero near-template only if review supports it. Do not substitute design-only counts for actual-source evidence.
- [x] Run static JSON/AST/hash inspection and explicit old-file preservation verification after final edits. All 480 future dynamic positions remain unrun; length checks remain pending.
- [x] Record completion without branch/commit/push. No dynamic, model, tokenizer or training work follows automatically.

## Verification

Before creating r1, the unchanged original auditor may run:
`python3 /private/tmp/lora-recalibration-v2-20261119/audit_data.py`.
After r1 exists, do not run that exclusion-based 154-file enumerator unchanged.
Use the frozen explicit list and compare original-manifest digests instead.

The parent will store a read-only metadata inspector at
`/private/tmp/lora-recalibration-v2-r1-20261119/audit_revision.py`.
It must read stored source only as strings/ASTs and report source/case/relation
counts, exact preservation, review bindings and false execution flags.
Its successful exit is not proof that any task function was dynamically run.

Plan self-review: scopes and schemas are consistent; all six sections of R3
and R1 preservation requirements map to the tasks above. The static-only user
boundary supersedes generic dynamic TDD, installation and commit templates.
