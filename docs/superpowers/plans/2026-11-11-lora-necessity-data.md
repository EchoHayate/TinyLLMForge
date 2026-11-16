# LoRA Necessity Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author and statically cross-review the approved 44 repair tasks, without executing their source.

**Architecture:** One offline JSON source holds specifications, source strings, literal cases and rationales. Main derives public/supervision/oracle views using explicit allowlists, binds bytes and performs static validation. Fresh reviewers check answer semantics and structure, not merely the JSON format.

**Tech Stack:** JSON and Markdown; existing Python 3.12 standard library for read-only JSON/AST inspection and deterministic serialization. No new runtime feature or production validator.

## Global Constraints

- Work in `/Users/bytedance/dev/TinyLLMForge` on existing `experiments/feedback-active-verification`; no checkout, worktree creation, commit, push, installation or network/remote activity.
- Use `apply_patch` for authored files. Preserve all previous tracked/untracked work.
- No model/tokenizer loading, training, source-string execution, importing task functions, eval/exec, candidate execution or old-root reuse.
- Structure specification: `docs/superpowers/specs/2026-11-11-lora-necessity-data-design.md`, approved by the user's subsequent “继续”.
- Counts are supervision 8, development 4, transfer 16, retention 16. Four balanced families per split; supervision and transfer family sets match.
- Every task needs at least two public cases, six different scoring cases, a public initial-bug witness, and two distinct error variants with scoring witnesses.
- Specifications and literal expected outcomes are authored before reference source; verification here is static reasoning, not executable TDD or dynamic correctness.
- No comments or one-letter identifiers in source strings.
- Keep benchmark_eligible, formal_evaluation_ready, net_benefit_established and isolated false.
- Existing `pilot_protocol.py` and `test_pilot_protocol.py` must remain unchanged.
- Separate model-input material from private scoring materials. Data split is not runtime isolation.

## Task 1: Author the 44-Task Offline Source

**Owner:** Main after three fresh author-agent startup failures; no other agent owns the file. Independent review remains required and cannot be replaced by main's self-review.

**Files:**
- Create: `experiments/lora_necessity/data/task_sources_v1.json`
- Report: `/tmp/tlf-lora-data-20261111-lW6uR7/task-1-report.md`

**Interfaces:** A JSON object with exact keys `kind`, `version`, `tasks`.
`kind="LORA_NECESSITY_TASK_SOURCE"`, integer `version=1`.
Each task has exactly:

`id`, `split`, `family`, `template_id`, `entry_point`, `spec`,
`initial_source`, `reference_source`, `public_cases`, `scoring_cases`,
`initial_witness`, `mutants`, `structure_notes`.

IDs, families and meaning match the structure specification. `template_id`
is ID plus `_structure_v1`. Task order is S01–S08, D01–D04, T01–T16, R01–R16.

`spec` is a concise English statement including complete input domains,
bounds and result semantics; explicit exceptions where applicable.
Each source string is one ordinary Python function with the same signature
for that task. No decorators, annotations, defaults, imports or external helpers.
Allowed builtins and container operations are ordinary pure Python; no I/O,
reflection or dynamic code. Source is data only and must not be executed.

Each case has exact keys `id`, `args`, `expected`, `reason`.
`args` is the positional argument list; expected is either
`{"kind":"return","value":<JSON value>}` or `{"kind":"raises","type":"ValueError"}`.
`reason` derives the expectation from the specification, not from running reference code.
Case IDs `P01`, `P02` etc. and `H01`...`H06` etc. are local to the task.
No duplicate argument lists within or between its two case sets.

`initial_witness` has exact keys `public_case_id`, `predicted_initial`, `reason`.
`predicted_initial` uses the same expected-outcome shape and is explicitly a manual
prediction, not an observed execution. It must differ from the public expected value.
At least one scoring case must also reject the initial source; name it in the
witness reason as `initial_scoring_case=Hxx` for a mechanically checkable link.

`mutants` is a list of exactly two objects with `id`, `source`, `scoring_case_id`,
`predicted_mutant`, `reason`; IDs `M01`, `M02`.
Each full source differs from reference, initial source and the other mutant.
Predicted outcome uses the same outcome schema; differs from its scoring case.
Describe the error mechanism and manual derivation. At least one case must
distinguish the two mutants from each other; explain this in `structure_notes`.
Avoid syntactically broken programs as "mutants"; use functional errors.

`structure_notes` records algorithm/control-flow distinction from related
supervision templates and any residual overlap risk. No claim of semantic
independence solely from ID/family/AST.

### Authoring Steps

- [x] Read Global Constraints above and all 44 table entries in the approved structure spec.
- [x] Check the target file does not yet exist. Record this initial missing-data state, not a failing runtime test.
- [x] Write literal specs, cases and manual reasons before source strings, using `apply_patch`; then add initial/reference/mutant functions. Temporary literal-data assembly and exact one-site string edits may avoid redundant transcription; every task/case remains individually authored. No generated Cartesian template bank or execution of source strings.
- [x] Check all JSON parses, IDs/counts/case links and signatures agree; parse source using `ast.parse` only. Do not call compile, exec, eval or import source.
- [x] Manually trace each stated public/variant witness and each expected case. If unsure, report the task ID and concern rather than inventing a pass.
- [x] Write report with exact files, counts, static commands/results, unresolved concerns and no-execution limitation. Return source ownership to main. No commit.

The main controller owns derived files and reviews. The author must not write
packaging code, review verdicts on behalf of others, or change the agreed families.

## Task 2: Derive Views and Static Integrity Evidence

**Owner:** Main; may prepare diagnostics while Task 1 is authored, but cannot
claim data completeness or assemble final views until Task 1 returns.

**Files:** `data/structure_v1.json`, `data/supervision_v1.json`,
`data/prompts_v1.json`, `data/oracles_v1.json`, `data/manifest_v1.json`
under `experiments/lora_necessity/`. Diagnostics live only in the exclusive
temporary root; no production builder is introduced.

- [x] Validate task source field sets, unique IDs, fixed schedule and family balance, source AST/signatures, nonempty specifications/reasons, case counts, unique JSON args and witness links. None of these checks proves semantic correctness.
- [x] Derive `structure_v1.json` from ID/split/family/template/structure_notes.
- [x] Derive supervision from S rows with only ID, entry_point, spec, initial_source, public_cases (without private reasons), reference_source and initial_witness. Never include scoring_cases/mutants/structure_notes.
- [x] Derive D/T/R prompts with only ID, entry_point, spec, initial_source, public_cases (without private reasons).
- [x] Derive D/T/R oracles with ID, reference_source, scoring_cases, initial_witness and mutants. Supervision-only scoring material stays in task_sources.
- [x] Produce `manifest_v1.json`: wrapper with `kind="LORA_NECESSITY_STATIC_DATA"`, `version=1`, `cpu_manifest`, `file_sha256`, `status="STATIC_ONLY"`, four false admission flags; task digests are SHA256 of canonical full task JSON, not identifier strings. File digests bind raw bytes of source and all four preceding derived views.
- [x] Reconstruct every derived view from explicit allowlists and compare exact values. Compare manifest task digests and file digests. Do not feed real outcomes to `score_synthetic`.

Serialization uses `json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2,
allow_nan=False) + "\\n"` for file bytes; task digest uses compact
`separators=(",", ":")` instead of indent, same other settings.
Views are lists of records in source order. `public_cases` contain only
`id`, `args`, `expected`; `initial_witness` retains its manual-prediction name.

## Task 3: Independent Static Reviews and Closeout

**Files:** `data/review_v1.json`, `DATA_AUDIT.md`, README and dedicated new
`.superpowers/sdd/lora-necessity-data-20261111-progress.md`.

- [x] Create full-addition review packages with current byte hashes, no empty tracked-only diff.
- [x] Fresh reviewers read specs, expected values and source strings; check all 44 tasks, all 128 S/T pairs, development overlap and primary retention labels. Do not run source or ask the author to certify its own answers.
- [x] Record task-spec and quality verdicts separately; refer structural changes to the user.
- [x] Resolve structural findings 1–3 after explicit approval of the replacement design. Completed in `2026-11-11-lora-data-structure-revision.md`; original CHANGES_REQUIRED snapshot remains archived, with no denominator reduction.
- [x] Preserve reviewer identity, input hashes, per-task outcome, concrete findings and pairwise structure evidence. Agent reviews are not human sign-off; no claim of different-model independence.
- [x] Review final packaging and admission flags. Review decisions bind source bytes; do not retain a PASS after the reviewed source changes.
- [x] Re-run static checks and the existing 26-test CPU suite, with exact output. Keep dynamic verification, tokenizer verification, human sign-off and runtime admission explicitly pending.
- [x] Write final audit and README status. No commit/push or real run. Stop with an honest static dataset result, or an explicit unresolved-review result if genuine design approval is needed.

Original closeout: the draft contained 354 cases after the independently reviewed
R10 correction and was CHANGES_REQUIRED. The subsequently approved three-task
revision is complete in `2026-11-11-lora-data-structure-revision.md`, with fresh
static review and unchanged case count. Dynamic correctness and pilot admission
remain unestablished; the original rejected snapshot is preserved.

## Plan Self-Review

All three tasks implement the approved static data stage. No runtime code is
added, so source-string examples are not executed under an alleged TDD cycle.
The normal test-plugin bootstrap has telemetry/install side effects; do not run
it under this local-only scope. Existing unittest regression runs need no plugin.
No worktree or dependency installation is needed for new data files in the
already selected experimental branch. Later production validators, tokenizer
and isolated dynamic execution require their own tested implementation.
