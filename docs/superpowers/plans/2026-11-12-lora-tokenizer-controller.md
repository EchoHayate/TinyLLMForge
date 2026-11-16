# Ordered Tokenizer Audit Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax; main owns this document and stage evidence.

**Goal:** Execute and independently reconstruct the complete ordered tokenizer audit without promoting provisional counts to real admission.

**Architecture:** A pure receipt/report module checks the frozen inventory, rolling chain and finalization. A separate controller internally owns one existing session and stops at the first failure. Output remains pending independent cleanup.

**Tech Stack:** Python standard library, unittest, synthetic tokenizer/import/metadata/file fixtures only.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge`, branch `experiments/feedback-active-verification`, HEAD `93b4e8e0dc5174db0426a5da9e29cc360ed331a7`.
- No SSH/network, installations, telemetry, real metadata discovery, real tokenizer/model loading, weights, GPU, training, candidate/task-source execution, commits, pushes, new branches or worktrees.
- The approved unittest suite may read frozen data as strings; this is not executing the contained programs.
- Use apply_patch, stdlib unittest/TDD, no inline code comments or one-letter identifiers. Preserve all pre-existing source/tests/data/documents and sealed evidence.
- Parent specification: `docs/superpowers/specs/2026-11-12-lora-tokenizer-admission-design.md`; detailed child design: `docs/superpowers/specs/2026-11-12-lora-tokenizer-controller-design.md`.
- Ordered rows: training 8, first 136, references 44, sampled_second 136. Input limit 4096; target including exactly one EOS limit 256. No truncation, deduplication, retry or denominator reduction.
- Five flags remain false: benchmark_eligible, formal_evaluation_ready, net_benefit_established, isolated, all_possible_second_rounds_validated.
- Main owns stage records in `/tmp/tlf-tokenizer-controller-20261112-2a6nok` and `.superpowers/sdd/lora-tokenizer-controller-20261112-progress.md`; agents own only assigned source/test pairs and fresh task evidence/report files.
- Per-task fresh implementers, independent task reviews, then independent whole-component review. Use complete untracked-addition diffs, not empty ordinary git diff.
- Existing checkout, no publication, local stdlib TDD and no telemetry override conflicting generic skill bootstrap/test-only/commit/worktree defaults. User already requested continuous local continuation; do not ask execution-mode questions.

## Scope Review

The approved parent already fixes architecture and acceptance. This child design resolves API/report detail within that scope, not a new experiment. Visual companion and new approval questions are unnecessary for this nonvisual continuation. Real admission and wrapper work are explicitly excluded.

### Task 1: Pure Inventory, Receipt and Report Verification

**Files:**
- Create `experiments/lora_necessity/tokenizer_audit_report.py`.
- Create `experiments/lora_necessity/test_tokenizer_audit_report.py`.

**Interfaces:**
- Consume `validate_profile(raw, approved_sha256)`, `PINS`, `REFERENCE_SHA256`, `parse_json`, `encoded`, `digest`, `FLAGS`, `fields`, `require`.
- Produce `validate_inventory(raw, approved_sha256) -> dict`.
- Produce `validate_record(row, record, run_id, profile_sha256, previous_chain, identity_sha256=None) -> dict`.
- Produce `build_report(inventory_raw, approved_inventory_sha256, profile_raw, approved_profile_sha256, records, finalization, failure=None, abort_failed=False) -> dict`.
- Report is exactly `{"public": public_summary, "private": private_evidence}`; all public/private fields and failure combinations follow child design sections 4–5.
- Public validation failures must raise ValueError with fixed text `tokenizer audit evidence invalid`, without private exception text.

- [x] **Step 1: Write focused failing tests.**

Read the child design completely. Use frozen `build_inventory` bytes, synthetic profile versions with existing FILE_PINS, and complete synthetic session receipts. Never call real inspect_runtime or open_session here.

Initial test scaffold:

```python
def test_complete_inventory(self):
    self.assertTrue(Path(__file__).with_name("tokenizer_audit_report.py").is_file())
    module = importlib.import_module("tokenizer_audit_report")
    actual = module.validate_inventory(self.inventory_raw, self.inventory_sha256)
    self.assertEqual(actual, self.inventory)
    self.assertEqual(len(actual["rows"]), 324)
```

Add cases for raw digest/noncanonical/duplicate-key/unknown-field rejection; data/reference pins; 324 order/row hashes/category shapes; primitive-safe IDs/messages/sample; independent copies. For per-record tests, preserve actual neutral/session schemas; test strict integers, empty IDs, prompt hash, messages/answer binding, arithmetic/prefix/tail, status-target consistency and chain continuity.

For build_report tests, construct complete successful and limit transcripts. Test all eight category edges 0/7/8/143/144/187/188/323, no observations after stop, missing/duplicated/reordered records, identity/run/profile mismatch, finalization count/chain/stop/runtime/snapshot/flags changes, unique EOS at target end, failure combinations, missing finalization, public privacy and aliases. Test partial success never attains admission.

- [x] **Step 2: Run RED and retain output.**

Run `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -m unittest discover -s experiments/lora_necessity -p 'test_tokenizer_audit_report.py' -v`.
The initial assertion must fail because the production file is absent, not because of test imports or invalid fixtures. Retain command, exit, output and source hashes in fresh `task-1-*` files.

- [x] **Step 3: Implement the pure validators and projector.**

Use strict field validation and canonical byte comparison before accepting inventories; reconstruct references from reference rows and validate existing pins. Keep raw-byte approval separate from canonical value digest. Use helpers for integer lists, digest text, flags and strict fields rather than copying validation blocks.

The per-record recurrence is:

```python
unsigned = {name: value for name, value in record.items() if name != "chain_sha256"}
expected_chain = digest({"previous": previous_chain, "record": unsigned})
require(record["chain_sha256"] == expected_chain, "receipt chain")
```

Reconstruct the full transcript from ordinal zero in build_report, not by trusting prior validator calls. A successful termination or declared infrastructure failure must account for the transcript ending. Project public rows from validated inventory and validated scalar measurements only. Never spread arbitrary record/identity/error mappings into public output. Preserve private records unchanged with independent copies. Do not perform I/O or load the runtime in this module.

- [x] **Step 4: Run GREEN, full suite once and self-review.**

Run focused command then `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -m unittest discover -s experiments/lora_necessity -p 'test_*.py' -v`.
Retain evidence and exact changed-file hashes; confirm no old files changed. Additional defects require regression RED/GREEN. Report in `task-1-report.md`; do not commit.

- [x] **Step 5: Independent task review and any repair/re-review.**

Main packages both full additions and bound hashes. Fresh reviewer inspects spec compliance and quality, returns evidence-referenced findings. Only approved task proceeds.

Task 1 execution checkpoint: complete. Independent Spec PASS / Quality APPROVED, findings `[]`; main 28 focused passes, verified 172-test full-suite receipt and exact full-addition hashes. Evidence: `task-1-review.md`, `main-task-1-tests.json`.

### Task 2: Owned Ordered Fail-Stop Controller

**Files:**
- Create `experiments/lora_necessity/tokenizer_audit.py`.
- Create `experiments/lora_necessity/test_tokenizer_audit.py`.

**Interfaces:**
- Consume Task 1 interfaces verbatim and `tokenizer_session.open_session(root, raw, approved_sha256)`.
- Produce `run_audit(root, profile_raw, approved_profile_sha256, inventory_raw, approved_inventory_sha256) -> dict`.
- Returns Task 1 envelope. Input/profile validation precedes session opening.
- Only existing internal session loading is allowed. No caller-supplied tokenizer/session/identity or retry/reset API.

- [x] **Step 1: Write focused failing tests.**

Use Task 1 helpers only when they do not import TestCase classes and duplicate test discovery. Synthetic sessions may implement measure/finish/abort; method call assertions must accompany actual full report assertions. Test full frozen 324-row success and exact full messages/answers, all category-edge limits, load failure, measurement failure, immediately invalid receipt, finalization failure, abort failure and cancellation. Assert no later measure/finish after protocol or infrastructure stop and no retry.

Add an integration case through the actual owned session with synthetic library/file/metadata fixtures, preserving real guard/session/counting behavior. This must not load installed third-party packages. Keep imported test modules qualified to avoid duplicate discovery.

Initial test scaffold:

```python
def test_controller_exists(self):
    self.assertTrue(Path(__file__).with_name("tokenizer_audit.py").is_file())
    module = importlib.import_module("tokenizer_audit")
    self.assertTrue(callable(module.run_audit))
```

- [x] **Step 2: Run RED and retain output.**

Run `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -m unittest discover -s experiments/lora_necessity -p 'test_tokenizer_audit.py' -v`.
Verify the absent implementation failure; retain all fresh `task-2-*` evidence.

- [x] **Step 3: Implement orchestration only.**

Prevalidate both raw inputs; open one session. Deep-copy each row's messages for the call so unexpected mutation cannot corrupt the approved denominator, and validate returned receipt against the original row immediately. Accumulate only validated copies. Break at first limit. Finish once only after complete or limit-stopped transcript; validate finalization through build_report. On normal errors classify OPEN/MEASURE/RECEIPT/FINALIZE, abort once when owned, and return a pending-cleanup failure envelope. Do not print errors or exception strings.

Compare each accepted neutral measurement state digest with the preceding accepted state before appending or measuring another row. Task 1 validates state continuity again at report reconstruction; this immediate controller comparison is additionally necessary to fail-stop on a changed state.

On cancellation, make one best-effort abort and re-raise the original BaseException even if cleanup fails. No session reset or environment restoration. Ensure finalization-validation failure follows the same FINALIZE path rather than leaking a raw ValueError. Pass only fixed failure dictionaries to report code.

- [x] **Step 4: Run GREEN, full suite once and self-review.**

Run focused command then the full approved command. Retain command/output/exit/source hashes, report `task-2-report.md`; no commits.

- [x] **Step 5: Independent task review and any repair/re-review.**

Main packages exact complete additions. Reviewer checks runtime ownership, fail-stop calls, cancellation and privacy against Task 1 APIs. Main resolves cross-task uncertainties.

## Final Integration Gate

Task 2 execution checkpoint: complete. Independent Spec PASS / Quality APPROVED, findings `[]`; implementer 19 focused and 191 full-suite passes, main fresh 191 full-suite passes in 15.308s. Cold-import/static/package/protection checks passed; whole-component review remains.

- [x] Main independently runs the full approved suite, binding exact source/log hashes and exit status.
- [x] Check syntax, no inline comments/one-letter identifiers, test uniqueness, no source/tests/pins lost.
- [x] Cold-import both new modules in a fresh Python process; prohibit third-party imports, runtime discovery, snapshot access, environment mutation and one-shot attempt consumption.
- [x] Mechanically reconstruct the four complete addition diffs and match current source hashes.
- [x] Independent total review of all four additions and integration evidence; resolve all important findings.
- [x] Main verify baseline preservation and Git branch/HEAD/clean tracked-index state. Record independent PASS/APPROVED and exclusive `LOCAL_TOKENIZER_AUDIT_CONTROLLER_ONLY_NOT_REAL_ADMISSION` receipt. No real-audit kind or cleanup attestation.

## Final Execution Checkpoint

Both tasks and the whole-component review are complete: Spec PASS / Quality APPROVED,
`final-findings.json=[]`. Main observed 191 unique tests passing in 15.308s, including
all 144 baseline tests and 47 additions. Cold-import, static and complete-diff
reconstruction checks bind the exact four source hashes; 404 protected files and
branch/HEAD/tracked-index state remain unchanged.

The final reviewer was independently dispatched by the parent using the explicitly
selected advertised route and is now closed. The child could not introspect its
model route. Parent dispatch evidence is in `review-dispatch-provenance.json`;
this is not independent underlying-model attestation. No new review is needed
merely because the child lacked tools to dispatch another reviewer.

Final receipt: `/tmp/tlf-tokenizer-controller-20261112-2a6nok/final-verification.json`.
This checkpoint is sealed only if that exclusive receipt exists and its source,
document and evidence hashes match. Do not rerun consumed helpers over accepted
outputs. No tokenizer/model loading, source execution, training, commits or push
occurred. The next distinct component is the process/resource/transfer wrapper;
real metadata approval, runtime compatibility, cleanup and scientific evaluation
remain separately gated.
