# LoRA Data Structure Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three rejected transfer repair structures and independently review the revised static bank.

**Architecture:** Preserve the entire prior data/review snapshot before replacing only T04/T09/T16 in the current source. Rebuild existing allowlisted views and byte/content bindings. Keep old judgments bound to old bytes and append fresh review evidence for changed tasks.

**Tech Stack:** JSON/Markdown and existing Python 3.12 standard library for data serialization, hashing, AST inspection and trusted CPU protocol tests.

## Global Constraints

- Work in `/Users/bytedance/dev/TinyLLMForge`, existing `experiments/feedback-active-verification`; no branch/worktree changes, commit, push, installation or network/remote work.
- Do not load a model/tokenizer, execute task source strings, import task functions, compile for execution, call eval/exec, train or run candidate/reference/mutant programs.
- Use `apply_patch` for authored changes. Temporary literal-data serialization and archival copies are allowed; preserve unrelated work and old evidence.
- Only T04/T09/T16 task objects change. Other 41 tasks, all eight supervision examples, 8/4/16/16 counts, families, arms, budgets and thresholds remain unchanged.
- Keep existing JSON schema version 1 and template IDs `ID_structure_v1`. These are schema/table identifiers, not a content-revision guarantee; new full-task hashes identify the revision.
- All four admission flags remain false: benchmark_eligible, formal_evaluation_ready, net_benefit_established, isolated.
- No task-function execution means new cases are authored data, not passed unit tests or executable TDD. Existing standard-library CPU regressions do not execute these functions. Do not bootstrap test plugins or telemetry.
- Independent assistant review is not human sign-off, runtime correctness, proof of independence or LoRA benefit.
- Parent input contract still applies: ordinary valid input lengths <=32, ASCII labels may be empty; no extra input-nonmutation scoring requirement.
- Evidence root: `/tmp/tlf-lora-revision-20261111-bI6CIM`.

## Task 1: Author Three Replacement Records

**Owner:** Fresh author agent, temporary output only. Main concurrently archives old data and prepares static binding checks.

**Files:**
- Create `/tmp/tlf-lora-revision-20261111-bI6CIM/replacements.json`
- Create `/tmp/tlf-lora-revision-20261111-bI6CIM/task-1-report.md`
- Optional literal-data author script only inside this evidence root.

**Interfaces:** `replacements.json` is a JSON list of exactly T04, T09, T16, in this order. Each row has the exact source schema:
`id`, `split`, `family`, `template_id`, `entry_point`, `spec`, `initial_source`, `reference_source`, `public_cases`, `scoring_cases`, `initial_witness`, `mutants`, `structure_notes`.
All splits are `transfer`; families are `bounds`, `multiplicity`, `order` respectively; template IDs unchanged. No repository edits by author.

- [x] Read approved `docs/superpowers/specs/2026-11-11-lora-data-structure-revision.md` and existing source schema in `experiments/lora_necessity/data/task_sources_v1.json`. Read eight S records for repair-context risks.
- [x] Author each precise English spec and at least two public/six scoring literal cases before function strings. Keep exactly two public/six scoring per replacement unless a concrete coverage omission needs more.
  - T04: capacity 1..32, start 0..capacity-1, length 0..capacity, non-bool integers; enumerate physical indices with wraparound. Cover capacity one, zero length, no wrap, exact-end boundary, crossing, full capacity and max capacity. Initial truncates at capacity or omits modulo.
  - T09: adjacent-distinct ASCII string labels, positive run counts 1..32, total expanded length <=32; skip/take 0..32 non-bool integers. Slice expanded elements and return runs with exact multiplicity; handle empty, zero take, beyond-end, partial first/last run, crossing, exact boundary and repeated nonadjacent labels. Initial counts each run as one element, ignoring count.
  - T16: <=32 non-bool integers with magnitude <=1000, negatives first and nonnegatives second, stable within each partition, duplicates retained. Cover empty, zero, all one sign, duplicates, mixed signs and displaced nonnegative order. Initial uses a swap partition that disturbs relative order; do not implement an initial that happens to preserve all tested order.
- [x] Cases have exactly `id`, `args`, `expected`, `reason`. IDs P01/P02 and H01..H06. `args` is positional argument list, expected is `{"kind":"return","value":...}`. Derive reasons from spec, never from executing source. Avoid repeated args within a task.
- [x] Add initial/reference/two mutant function strings: ordinary pure Python, same signature per task, no imports/defaults/decorators/annotations/comments/one-letter identifiers. Mutants must be four distinct full strings with meaningful different errors, not syntax defects.
- [x] `initial_witness` has `public_case_id`, `predicted_initial`, `reason`; public prediction differs from expected and reason names a scoring witness as `initial_scoring_case=Hxx`. Each mutant has `id` M01/M02, `source`, `scoring_case_id`, `predicted_mutant`, `reason`; prediction differs from expected. `structure_notes` explains a same-input discriminator between mutants, its two predictions and concrete supervision differences/remaining overlap.
- [x] Statically inspect JSON/AST/signatures and manually trace every expected outcome, initial witness and mutant prediction. No execution of embedded programs or ad hoc interpreter. Main runs whole-bank schema checks after merge.
- [x] Write report: DONE or DONE_WITH_CONCERNS, exact files/hashes/counts, manual reasoning, unresolved concerns, no-execution statement; return ownership. No commit.

## Task 2: Integrate and Independently Review

**Owner:** Main binds data; fresh reviewer checks semantics and structural separation.

**Files:**
- Archive all seven current JSON files under `experiments/lora_necessity/data/history/2026-11-11-before-structure-revision/`, plus old audit text.
- Modify only current source, structure, prompts, oracles, manifest and review JSON; supervision bytes remain unchanged.
- Update the three structure-table rows in `docs/superpowers/specs/2026-11-11-lora-necessity-data-design.md`.
- Temporary revised-source diff and static inspection evidence under the new evidence root.

- [x] Verify old source `f198764ec17690a7c9e3b289f904a166f77727478212381bb3aaef584efbf8d2`, old review `31402c530a1f36be7d09118bc6465c0a02b84ad8018743f7bc57a4de19440184`; archive bytes exclusively before edits. Preserve previous independent report embedded in old review.
- [x] Validate replacement list IDs and exact unchanged fields, replace only those records, serialize sorted ASCII/indent2/newline. Assert canonical equality for every other record and identical supervision bytes. Make the current review explicitly pending as soon as source changes; never leave an old PASS attached to new bytes.
- [x] Rebuild structure/supervision/prompts/oracles using the existing explicit allowlists and CPU manifest builder. Recalculate raw-file and full-task hashes. Do not modify frozen `pilot_protocol.py` or `test_pilot_protocol.py`.
- [x] Run a copied, unchanged static checker against the revised package:
  `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 /tmp/tlf-lora-revision-20261111-bI6CIM/check_data.py --check-package`
  Expected: 44 tasks, 176 sources, 88 public/266 scoring cases, 88 mutants, all false flags; extra cases only if justified.
- [x] Generate actual old/new full source diff. Fresh reviewer reads task brief/report/diff, approved revision, archived original report and eight S tasks. Independently check all new outcomes/predictions, 24 changed S/T pairs, D/R and remaining T overlaps, closing or retaining each old finding with reasons. Do not rerun task programs or trust author predictions.
- [x] Fresh final packaging review checks unchanged 41 tasks, unchanged supervision/config/families, exact projections/hashes/false flags, review evidence composition and documentation boundaries. No review reruns of passed suites.
- [x] Check whether findings require another design decision: no new important finding remained. Two author spec inconsistencies were corrected within the parent contract before independent review; no further task replacement.

## Task 3: Close Out Static Revision

**Files:** `experiments/lora_necessity/{DATA_AUDIT.md,README.md,data/review_v1.json}`, dedicated `.superpowers/sdd/lora-data-structure-revision-20261111-progress.md`.

- [x] Embed complete new independent report and hash in current review JSON. Bind old report via the durable archived review. Preserve all 128 pair judgments: 104 inherited only after proving unchanged S/T task objects, 24 freshly reviewed. Keep old rejected judgments in history.
- [x] Record exact scope of each verdict, all unresolved findings and runtime gates. A clean static review cannot turn any admission flag true.
- [x] Run trusted CPU suite:
  `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -m unittest discover -s experiments/lora_necessity -p 'test_*.py' -q`
  Expected: 26 passing tests, no task-source execution.
- [x] Validate current and archived report/file hashes, all 44 task rows and 128 unique pairs, unchanged configuration and supervision, no tracked/index changes.
- [x] Update audit/README/ledger and final user summary. Do not initiate dynamic correctness, tokenizer or training work under this approval.

Completed 2026-11-11: `STATIC_REVIEWED_NOT_RUNTIME_VALIDATED`. Fresh structural
and final package reviews passed; all three old structural findings closed.
Final CPU regression: 26 tests, 0.103s. Current review/source hashes and exact
evidence are recorded in the dedicated revision ledger and `DATA_AUDIT.md`.

## Self-Review

This is a data-only revision within the approved three structures. Ordinary
template/version labels are kept for schema compatibility and explicitly
distinguished from content hashes. Cases are not executed; no runtime code,
new optimizer, harness or validation security boundary is introduced.
