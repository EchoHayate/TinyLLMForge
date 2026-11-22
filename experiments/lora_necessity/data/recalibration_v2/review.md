# LoRA recalibration v2 — static data review

Date: 2026-11-19. Status: **static review complete; structural acceptance blocked**.

## Scope

This directory materializes the approved twelve-task development design, not
an experiment result. It does not implement the v2 model driver or launch any
source, model, tokenizer, training or remote process.

The original v1 data and consumed runs remain unchanged. New C01–C12 material
is development-only, not LoRA training data or an independent final evaluation.
No old transfer/retention answers or model outcomes were used to select cases.
Known design-level structural neighbors are disclosed, not renamed away.

## Authorship

- Parent controller: 96 case inputs, manually derived expected values, coverage
  notes and 210 structural comparisons.
- Independent reference B: Huygens,
  `01a1f26b-d5cc-73c2-94fe-0d5d36bb1412`. Read only the approved design and
  source-free cases before authoring. B file was sealed before primary A existed:
  `4fd8137a14c2b6fd1495b84636913b5508ce20699e0cb29e0ee768ef59084e45`.
- Primary/reference A and incorrect-source author: Kant,
  `01a1f270-b7c8-78b2-b2cb-4f640b298e66`. Primary sources were sealed before
  reading B for byte-preserving integration:
  `a55695cf0d8fb8d8a1d49deec00a56ae0e0a60e375314d77e3b294b9f29729ef`.
- Independent case/B reviewer: Franklin,
  `01a1f26c-ae6f-7e83-82bf-9ce2044047b7`.

These are separate assistant authors/reviewers, not independent human sign-off.
Different source text alone is not evidence of independent authorship; the
read restrictions and before-integration source seals are recorded separately.

## Handwritten cases

The case inventory contains 24 approved public examples and 72 private examples.
Every task has P01/P02/H01–H06 and eight distinct full argument vectors.
All have legal-input descriptions, coverage and manual derivations.

Independent first review manually traced all 96 expected values and found no
incorrect oracle value or illegal input. It found one derivation-only issue:

- R1, C01/P01: the prose listed final occupancy as 2, but the final accepted
  request has size 1 after expiry. Corrected trajectory: 2,2,2,2,1.
  Public input and expected booleans are unchanged. The original finding is
  retained in the private review, not silently relabeled as a clean first pass.
  Franklin independently verified the corrected field and unchanged remaining
  case data, then closed R1. The case-only review is now approved.

Original case artifact SHA256:
`c0505a26e52380b4e0923dd3ae5ce4d88ec453e1b68d7d4ad614a29c7862bb5e`.
Corrected artifact SHA256:
`220f30b8b1d8c3720bfb7374ff9fa913669241a0a65f8d9b278f7573747bfd91`.
Reference B was authored using the original case prose; inputs and all expected
outputs were unchanged by this correction.

## Structure and minimum repair

`structure.json` explicitly enumerates 96 new-task/supervision, 48
new-task/old-development and 66 new-task/internal relationships. Classifications
distinguish different structure, shared mechanisms and a near-template concern.
Enumeration is not proof of independence or novelty.

C01:S01 is a confirmed near-template blocker: the initial source only requires
`reservation[1] >= time` to become `reservation[1] > time`. S01 likewise removes
equality from time membership. All surrounding capacity and accepted-only
accounting is already correct. The richer correct program adds no actual repair.

C04:S03 is a second confirmed near-template blocker. Replacing begin-time
`totals.setdefault(transaction, 0)` with `totals[transaction] = 0` repairs the
observable behavior for all legal inputs. Terminated IDs cannot be used until
a new begin, which overwrites stale totals. Thus explicit removal of inactive
entries, rollback bookkeeping and additional interleaving logic are not needed
in the minimum patch. Like S03, it only resets a cumulative slot at a new segment
boundary. The parent checked the actual C04/S03 sources before accepting this
finding and changed its classification from shared mechanism to near template.

C10 also retains a limitation: an eviction trace cannot be reconstructed from
only final last-occurrence order, but per-prefix recency reconstruction is still
a valid solution. Neither changed function names nor output wrappers establish
unseenness; the concrete refresh/eviction repair is the relevant unit.
The independent reviewer found no additional definite C10 blocker against the
authorized S/D comparisons, but did not compare unread old transfer/retention
source. The trace catches a duplicate-key variant whose final keys alone happen
to be correct; that does not make the recency mechanism novel.

Franklin independently reviewed all 210 actual-source relationships. Final
classification counts are 2 near-template, 59 shared-mechanism and 149
different-structure. These labels are bounded static judgments, not statistical
independence or pretraining-novelty claims. No smaller, automatically admitted
ten-task substitute has been created.

## Static checks versus unperformed checks

Parent JSON checks verified twelve task IDs and four groups of three; all 24
public examples match the approved values. Explicit domain inspection checked
all 96 inputs, including transaction-lifecycle preconditions, strict integer
types, label grammar and per-task aggregate lengths.

Reference-B author checks and parent seal/signature verification passed for
twelve standalone functions. Franklin's independent source review approved all
twelve B functions and manually mapped all 96 cases without disagreement.
No stored source was executed.

The parent also read all twelve final public specifications and checked their
legal domains, return contracts and public-only content against the approved
design. Exact case projection and source sealing remain separate metadata checks.

The parent ran the finalized-data inspector successfully: all 60 ASTs, twelve
public/oracle projections, 36 witness bindings, 210 relation identities, exact
primary/B source seals, all revised case objects and the 154-file preservation
digest passed. Independent review additionally read all 48 primary sources,
hand-traced all 36 designated incorrect witnesses and checked twelve complete
public prompts. It found no reference, witness or public/private projection
error, reusing the earlier 96-case and 12-reference-B reviews. Its overall
acceptance is BLOCKED because of the two structural findings, not source syntax
or oracle disagreement. Neither a syntactic parse nor a manually predicted
wrong outcome is a dynamic rejection witness.
Unknown validation states remain pending, not inherited from v1.

Future dynamic data checking has 12 tasks × 5 programs × 8 cases = 480 planned
positions. This is separate from the model diagnostic's 72 generations and
at most 624 candidate executions. No such dynamic positions were run here.
Tokenizer checks for 4096 input/256 output tokens also remain pending.

## Evidence location and preservation

Private authoring/review evidence:
`/private/tmp/lora-recalibration-v2-20261119/`.
The private inspection script `audit_data.py` only reads JSON, parses ASTs and
checks identities/hashes. It does not import or invoke stored task functions.

The preexisting LoRA tree contained 154 files excluding `__pycache__`; sorted
relative paths, NUL and each file's binary SHA256 produce aggregate SHA256
`e9b4510aa7247fe926764d6ff43a4d1b046fd09fa8ad19c5680243ee1137792e`.
Final preservation comparison excludes this new dataset and its new result
report, but includes all prior LoRA reports and data.

## Findings and disposition

| Finding | Disposition |
| --- | --- |
| R1: C01/P01 final occupancy prose | Corrected and independently closed; all case arguments and expected values unchanged. |
| F01: C01:S01 time-boundary minimum repair | Confirmed and open; no initial-source redesign authorized or performed. |
| F02: C04:S03 accumulator-reset minimum repair | Classification corrected to near-template; underlying structural blocker remains open. |

The approved design and all 60 authored source strings remain unchanged after
review. Correcting classification is not a repair of the research limitation.
Any subsequent initial-defect revision needs its own recorded design decision,
new source identities, hand-derived witnesses and renewed structural review.
It must not quietly add extra bugs to make a problem appear harder.

Independent complete review:
`/private/tmp/lora-recalibration-v2-20261119/full-review/independent-review.md`,
SHA256 `23be8acbfdd4deac1ebaaf2c80123212d9aa3fb8016fbe3cb17837f73ffe407f`.
Original findings:
`/private/tmp/lora-recalibration-v2-20261119/full-review/findings.json`,
SHA256 `ecf1388cde9f3de103c982d1be52cc6c9a09a4de48105d3f98c5164f614cd7b5`.
These reviews refer to structure snapshot
`7195598cfae7bc9f30fdda30af0a2abe6cbbfc5b4c04d8e079fce05c98449cf5`;
the final structure incorporates F02 and binds that provenance explicitly.

`manifest.json` binds this report and the three JSON data files. Its checksum
bindings preserve this blocked review snapshot, not an execution-ready freeze.
Dynamic and tokenizer-length states remain pending. Run authorization,
benchmark eligibility, formal evaluation readiness, net benefit and runtime
isolation remain false. No static result grants model-run permission or
establishes LoRA gain.
