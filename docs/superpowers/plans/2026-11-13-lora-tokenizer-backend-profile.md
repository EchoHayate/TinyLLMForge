# Tokenizer Backend Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to execute this dependency-ordered plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Obtain and review an evidence-bound, concrete backend profile before implementing a real tokenizer launcher.

**Architecture:** Resolve the exact runtime, resource ownership, storage accounting and remote lifecycle mechanism from narrowly scoped evidence. Produce either a reviewed profile or explicit missing prerequisites; neither result claims resource enforcement. This is the first operational implementation plan, not a complete launcher coding plan.

**Tech Stack:** Existing Git checkout, Python 3.12 standard library for local evidence verification, Markdown/JSON records, and separately authorized read-only metadata collection.

## Global Constraints

- Binding spec: `docs/superpowers/specs/2026-11-13-lora-tokenizer-backend-design.md`.
- Existing checkout only; no branch/worktree, commit, push or memory update.
- Preserve all accepted source/tests/data/documents and old evidence roots; use `apply_patch` for edits and exclusive creation for generated evidence.
- 4 GiB = 4,294,967,296 bytes aggregate memory; at most 8 process/thread tasks.
- Peak occupancy of all new task files is at most 128 MiB = 134,217,728 bytes, including distinct producer/receiver copies.
- Result at most 64 MiB = 67,108,864 bytes; stdout/stderr each at most 1 MiB = 1,048,576 bytes.
- Work at most 540 seconds and total at most 600 seconds from the original connection claim; cleanup at most 60 seconds within the original total window.
- Single-thread configuration; core dumps disabled; one connection and one audit per authorized launch; no automatic retry.
- This plan does not create containers, configure resources, load model/tokenizer libraries, transfer private answers/token IDs, train or use GPUs.
- A written design approval is not remote-operation authorization.
- A metadata declaration is not enforcement evidence. No new boolean-only admission class, fake launcher or mock-result promotion.
- Private evidence directories 0700, private files 0600. Do not include credentials, environment contents, other workloads or private task data in reports.

## Scope And File Map

This plan produces a real-environment dossier, not production Python changes.
No production code steps or tests pretending to validate a container appear here.
The later launcher and lifecycle coding plan depends on this dossier and cannot
honestly be specified against invented image IDs or platform APIs.

| File | Responsibility |
| --- | --- |
| `.superpowers/sdd/lora-tokenizer-backend-20261113-progress.md` | Phase ledger and pointers; append new progress, preserve chronological states |
| `experiments/lora_necessity/20261113-backend-profile-query-scope.md` | Exact proposed read-only query, selectors, limits and authorization status |
| `experiments/lora_necessity/20261113-backend-profile-review.md` | Public, non-sensitive profile review and missing prerequisites |
| Fresh private evidence root: `baseline-hashes.json` | Entry source/evidence identity |
| Fresh private evidence root: `scope.json`, `receipt.json`, `stdout.bin`, `stderr.bin` | Authorized query record; never reuse a previous query root |
| Fresh private evidence root: `profile.json` | Evidence-linked runtime/resource/storage/lifecycle facts and missing fields |
| Fresh private evidence root: `verification.json` | Final binding of evidence and new documents |

The dated scope/review paths are reserved for this plan and must not overwrite an
existing file. If they already exist on execution, read and reconcile them first;
do not assume this plan remains unfinished. Do not create empty marker files now.

## Task 1: Reconcile Evidence And Selectors Locally

**Files:** Read the binding spec, latest assessment and existing seals; append the
phase ledger only after completing work. Create the fresh private baseline record.

**Interfaces:**
- Consumes the current Git identity and existing evidence hash maps.
- Produces `BACKEND_PROFILE_EVIDENCE` pointing to a fresh private root and a list
  of exact available selectors or `MISSING_IMAGE_ID`/`MISSING_LIFECYCLE_PROVIDER`.
- No SSH call, runtime import or image enumeration.

- [ ] Confirm branch/HEAD and preserve dirty/untracked work:

  ```bash
  git branch --show-current
  git rev-parse HEAD
  git diff --name-only
  git diff --cached --name-only
  ```

  Expected entry identity is `experiments/feedback-active-verification` at
  `93b4e8e0dc5174db0426a5da9e29cc360ed331a7`, with clean tracked/index diffs.
  Changed identity requires reconciliation, not reset or checkout.

- [ ] Create one private root and verify protected evidence before new work:

  ```bash
  umask 077
  export BACKEND_PROFILE_EVIDENCE
  BACKEND_PROFILE_EVIDENCE=$(mktemp -d /tmp/tlf-tokenizer-backend-profile-XXXXXXXX) || exit 1
  PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 - <<'PY'
  import hashlib
  import json
  import os
  from pathlib import Path

  root = Path(os.environ["BACKEND_PROFILE_EVIDENCE"])
  prior = Path("/tmp/tlf-tokenizer-backend-design-20261113-26nlsn63")
  seal = json.loads((prior / "verification.json").read_text())
  protected = json.loads((prior / "baseline-hashes.json").read_text())
  for name, expected in seal["evidence_sha256"].items():
      path = prior / name
      assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)
      protected[str(path)] = expected
  for name, expected in seal["documents_sha256"].items():
      path = Path.cwd() / name
      assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)
      protected[str(path)] = expected
  protected[str(prior / "verification.json")] = hashlib.sha256(
      (prior / "verification.json").read_bytes()
  ).hexdigest()
  for name, expected in protected.items():
      assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
  assert root.stat().st_mode & 0o777 == 0o700
  with (root / "baseline-hashes.json").open("x") as output:
      json.dump(protected, output, indent=2, sort_keys=True)
      output.write("\n")
  print("ENTRY_PRESERVATION_VERIFIED", len(protected))
  PY
  ```

  Expected: one verification line; no remote side effect. Missing or changed
  evidence stops the step. Later ledger appends are scoped exceptions recorded
  with before/after hashes, not blanket permission to rewrite prior entries.

- [ ] Read the latest capability/research evidence and the historical image section:

  ```bash
  cat experiments/lora_necessity/20261113-readonly-capability-success.md
  cat experiments/lora_necessity/20261113-container-backend-assessment.md
  sed -n '102,117p' experiments/feedback_adaptation/20261105-remote-isolation-preflight.md
  ```

  Known starting result: no verified Python 3.12 candidate image ID and no
  admitted lifecycle provider are supplied by these records. Historical
  interpreter paths are not a container image. Do not invent an image tag.

- [ ] Record selectors found in task-owned evidence only. For each, retain exact
  value, provenance, observation time and permitted query. If no exact candidate
  exists, mark the corresponding missing prerequisite; request a task-approved
  image/provider reference rather than scanning unrelated workloads.

**Task exit:** Preserved evidence plus a selector/gap inventory. This is useful
even when profile discovery cannot yet proceed, but is not a reviewed profile.

## Task 2: Freeze A Narrow Metadata Query Before Connecting

**Files:** Create `experiments/lora_necessity/20261113-backend-profile-query-scope.md`;
create query evidence only if the exact query subsequently receives authorization.

**Interfaces:**
- Consumes Task 1 selectors and missing-prerequisite inventory.
- Produces an exact `PROPOSED_READONLY_PROFILE_QUERY`, then either a separately
  authorized single-query receipt or `NOT_EXECUTED_AUTHORIZATION_MISSING`.
- An absent image/provider selector never becomes permission for a broad listing.

- [ ] Write the scope with these concrete sections: purpose; exact existing target
  and account; selector provenance; exact read-only command/argv; allowed returned
  fields; time/output limits; local private evidence destination; forbidden
  operations; stop conditions; authorization status.
  Do not provide a runnable command until all its selectors are real and reviewed.
  Missing selectors are a recorded stop, not a string to substitute casually.

- [ ] Limit image metadata to the specified image's ID/digests, OS/architecture,
  declared user/entrypoint/command, declared volumes and relevant runtime
  requirements. Exclude image/container environment values, histories, layers,
  other containers and other images. No image pull, export, create or start.
  Reading a command is metadata inspection, not permission to execute it.

- [ ] Limit any host-runtime investigation to the explicitly selected interpreter
  and dependency metadata paths. It requires its own listed commands; no import
  of transformers/tokenizers, no model file read and no automatic interpreter
  execution. Metadata alone cannot prove the resulting image is compatible.

- [ ] Specify the proposed query limits: one connection to the existing target,
  local work 55 seconds, total 60 seconds, remote command 35 seconds plus at most
  2 seconds termination grace, stdout/stderr each 1,048,576 bytes.
  Retain strict host-key checks, no credential delegation/agent forwarding,
  no connection reuse, no interactive fallback and no automatic retry.
  The old capability helper is a construction reference, not a callable
  profile collector; never execute it in its sealed directory.

- [ ] Obtain approval of the exact query scope before preparing/executing a
  new operational helper. The approved query and helper must be inspected
  before connection, stored with their hashes, and locally checked without
  network activity for argv/proxy construction and output/time boundaries.
  Concrete helper implementation is a separate short execution step once
  its real query exists, not speculative code in this plan.

- [ ] If approved, execute once in a new private query root and inspect individual
  command statuses, identity and completion markers. Preserve failure as failure.
  No root reuse, retries, substitute targets or security-setting changes.
  If not approved or missing selectors, record the exact unsatisfied gate and
  stop without making a connection.

**Task exit:** An authorized metadata receipt with bounded raw evidence, or an
explicit no-execution outcome. Neither permits configuration or task creation.

## Task 3: Assemble And Review The Concrete Profile

**Files:** Create private `profile.json` and
`experiments/lora_necessity/20261113-backend-profile-review.md`.

**Interfaces:**
- Consumes exact Task 1/2 evidence and the spec's sections 5–8.
- Produces a profile with all fields below; unresolved fields appear in
  `missing_requirements`, never as guessed defaults.
- No API returns a capability merely because a user supplies booleans.

- [ ] Create an evidence-linked profile with these required fields:

  | Field | Required content |
  | --- | --- |
  | `kind` | `BACKEND_PROFILE_OBSERVATIONS` |
  | `target_identity` | Observed host/UID and raw evidence reference |
  | `runtime` | Exact candidate image identity and metadata, interpreter and dependency requirements, observed versus unobserved distinctions |
  | `resource_domains` | Every new helper/worker/control task, group ownership and aggregate-limit allocation or an explicit missing allocation |
  | `storage_domains` | Every new file destination and copy, limit mechanism, byte/inode accounting and total budget allocation or explicit missing coverage |
  | `lifecycle` | Exact available control mechanism, supported object identity, deadline mapping and cleanup observations or explicit missing mechanism |
  | `authorization_refs` | What each recorded authorization actually permits; no inferred launch permission |
  | `evidence_refs` | Private source filename, byte hash, capture time and command/receipt binding |
  | `missing_requirements` | Ordered list of unsatisfied spec requirements with the evidence/actor needed to resolve each |

  An empty or absent mandatory fact belongs in `missing_requirements`; it cannot
  be removed because a service declares generic support.

- [ ] Reconcile storage across both ends. Prefer an output design avoiding two
  simultaneous full result files, but do not assume the existing reader already
  provides that design. Budget each new copy and each auxiliary mount.
  If logical output size, allocated blocks or metadata costs have different
  enforcement mechanisms, record each and the conservative combined bound.
  Do not convert the peak-occupancy decision into a cumulative-I/O requirement.

- [ ] Reconcile resource ownership and time. Show how all new helper processes
  fit within aggregate limits, what platform control plane is outside them,
  and why that boundary is permitted. Require a concrete independent remote
  expiry mechanism and conservative cross-clock budget mapping.
  A Docker stop grace value or local deadline counter does not fill this field.

- [ ] Write the public review with exactly these conclusions:
  available facts; missing facts; proposed exact configuration; B01–B10 coverage;
  remaining authorizations; next responsible actor. Omit private paths/contents
  where unnecessary. If any required fact is missing, use `PROFILE_INCOMPLETE`
  and list it; do not declare the launcher ready to implement against unknown APIs.

- [ ] Obtain review of the profile. Only an evidence-complete and approved profile
  may be labeled `BACKEND_PROFILE_REVIEWED_NOT_ENFORCED`.
  Enforcement, complete cleanup, real tokenizer and scientific-admission flags
  remain false in either outcome.

**Task exit:** Reviewed profile or actionable prerequisite gap report, with no
container or tokenizer execution.

## Task 4: Preserve The Result And Hand Off The Next Phase

**Files:** Append the phase ledger, create private `verification.json`.

**Interfaces:**
- Consumes all new records and the entry baseline.
- Produces a source/evidence-bound phase outcome and an explicit next-phase scope.

- [ ] Check new docs for missing values disguised as readiness, inconsistent
  counts, unfrozen image tags, omitted helper/write paths, reset deadlines and
  unapproved external operations. Correct only the current phase's documents.
- [ ] Reverify the complete baseline, allowing only the explicitly recorded ledger
  append; retain before/after hashes and verify its prior content remains intact.
  Bind every new receipt, raw output, profile and report to the new final seal.
  Preserve all accepted source/tests and frozen data.
- [ ] Record actual commands and outcomes. Do not run or count component tests
  merely to dress a documentation-only phase as new implementation evidence.
  If a later operational helper is modified, validate that helper specifically;
  its checks are not backend behavior tests.
- [ ] If the profile is incomplete, stop with the concrete external prerequisite.
  If complete and reviewed, use `writing-plans` for the next, separately scoped
  launcher/lifecycle implementation plan with exact files, full code/TDD steps,
  local validation and ownership/error cases derived from this profile.
  The implementation plan itself is not permission for remote behavior tests.
- [ ] Present completion status and the execution choice for the next phase:
  inline checkpointed execution is recommended for the sequential resource
  ownership work; fresh subagents are an alternative only for independent tasks.
  Never resume historical sessions or restart completed agents.

## Spec Coverage And Phase Boundary

| Spec requirement | This plan | Subsequent required phase |
| --- | --- | --- |
| §§1–3 scope, authorization, budget | Tasks 1–3 | All later phases preserve them |
| §§4–5 identity and component ownership | Tasks 1–3 resolve real inputs | Launcher implementation after review |
| §6 resource and storage enforcement | Task 3 designs the concrete allocation | Implement and behavior-test actual backend |
| §7 expiry and cleanup | Task 3 resolves the provider and timing design | Implement, then test disconnect/failure paths |
| §8 B01–B10 | Task 3 records coverage/missing proof per row | Independent bounded behavior validation |
| §9 staged conclusions | Task 4 | Separate real-tokenizer launch admission |

No production backend source file, Docker configuration change or remote
execution is part of approving this written plan. Its checkboxes start unchecked.
