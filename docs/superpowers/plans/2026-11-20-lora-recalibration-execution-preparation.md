# Recalibration Execution Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the exact 15 fixed-source bundles and their local dependency identities as a non-executable preparation artifact.

**Architecture:** This is the first independently testable slice of the approved fixed-execution design, not its complete launcher. Reuse the existing pure batch builder, retain a public binding manifest plus private bundle bytes encoded as hex, and validate supplied preparations by reconstructing them from original inputs. Real launch configuration, preflight admission, full execution-source closure, collection and remote execution remain outside this subplan.

**Tech Stack:** Existing Python standard library, unittest, immutable bytes inputs, canonical ASCII JSON and SHA256; no new dependencies.

## Global Constraints

- Design: `docs/superpowers/specs/2026-11-20-lora-recalibration-fixed-execution-design.md`, SHA256 `88f69b71d60fc7165d4e67fa739f8a3a7e743ee0456551a1adf6d7e1ecd905c3`.
- 固定 12 题，每题两参考、三错误源码、八用例；共 480 项、15 批、每批 32 项。参考分母 192，具名错误见证分母 36。
- 本地只允许 JSON/AST、纯组件及人工合成回执；题库源码不得在宿主执行。
- 不更改冻结数据、旧源码、六个已完成新版纯组件、评分或 token 预算。
- 不安装、遥测、切分支/worktree、提交/push、清理旧文件或恢复旧 session。
- No SSH, network, real tokenizer/model loading, generation or training. The existing approval applies to local implementation, not real launch.
- No invocation/import of `validation_remote` or `validation_task_worker`; these are bytes to pin, not callable dependencies.
- No filesystem, subprocess, clock, random-ID or environment access in the new production module. All input bytes and identities are explicit.
- Stage and 15 distinct batch IDs are strict built-in strings, exactly 32 lowercase hex; stage ID cannot equal any batch ID.
- Source/request/worker/bundle limits stay 4096/8192/16384/131072 bytes via the existing batch builder.
- Supplied source bytes are checked snapshots, not proof of the modules executing in the current interpreter.
- Existing six pure-module pins are a dependency subset only. `execution_source_manifest_complete` is always false; this builder neither claims nor accepts the future complete execution-source manifest.
- Output `status=PREPARATION_ONLY`; all readiness, provenance, authorization and research flags are false. No caller flag, prebuilt PASS, profile, preflight or approval argument upgrades them.
- Test with existing unittest and bytecode disabled. Do not use networking bootstrap/flush/telemetry from test skill templates.
- Use `apply_patch`; no inline code comments or one-letter variable names. Preserve concurrent work and untracked files.

---

### Task 1: Pure Preparation Builder And Reconstruction Validator

**Files:**
- Create: `experiments/lora_necessity/recalibration_execution_package.py`
- Create: `experiments/lora_necessity/test_recalibration_execution_package.py`
- Parent owns plan progress, stage result note and `.superpowers/sdd/lora-recalibration-validation-adapter-20261119.md`; implementer must not edit these.

**Interfaces:**
- Consumes `recalibration_validation.build_batches(raw_files, worker_raw, run_ids)` and `WORKER_SHA256`; `raw_files` is the existing six-file byte mapping.
- Consumes `pilot_protocol.FLAGS/fields/require`, `recalibration_data_contract.FILE_PINS/_plain`, `validation_contract.digest/encoded`.
- Produces `build_preparation(raw_files, worker_raw, backend_raw, source_raw, *, stage_id, run_ids) -> dict`.
- Produces `validate_preparation(preparation, raw_files, worker_raw, backend_raw, source_raw, *, stage_id, run_ids) -> None`; invalid inputs or unequal reconstruction raise `ValueError`.
- `source_raw` has exactly the six names in `SOURCE_PINS` below, each value strict bytes, at most 1048576 bytes, exact pin. Wrong/missing/extra keys or equal-valued key subclasses are rejected.
- `backend_raw` is strict bytes at most 1048576 bytes and must match `BACKEND_SHA256`.
- Output is JSON-native: outer keys exactly `public/private`; private keys exactly `bundles_hex`, a 15-element list of exact canonical raw bundle bytes encoded as lowercase hex.
- Public manifest is exactly the dictionary built below plus its `preparation_sha256=digest(unsigned_public)`. This digest binds the bundles through their byte digests; validation also compares actual private bytes.
- The output carries no complete plan rows, raw oracles, predictions, roles, source-module bodies, runtime profile or shell commands.
- Changing an input ID is permitted only as a distinct unexecuted preparation. It changes the binding and grants no execution or retry permission; claims are not implemented here.
- Constants/field names below are the interface contract. Candidate code is a starting point, not a reason to dismiss a correctness finding.

**Scope and global constraints:** All Global Constraints above apply verbatim. This task deliberately implements the non-executable preparation portion of the package responsibility; live manifest admission and the other three modules are not deliverables of this subplan.

- [x] **Step 1: Write failing tests before production implementation**

Use existing data only as JSON/AST bytes. Import test modules, not their TestCase classes, if any fixture reuse is needed; avoid accidental double suite discovery.
Initial discoverable assertions must fail because the module/behavior is missing rather than an unhandled import error:

```python
import importlib.util
from pathlib import Path
import unittest

from recalibration_data_contract import FILE_PINS


if importlib.util.find_spec("recalibration_execution_package") is not None:
    import recalibration_execution_package as package
else:
    package = None


ROOT = Path(__file__).resolve().parent


class ExecutionPreparationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(package, "execution preparation not implemented")
        self.raw_files = {
            name: (ROOT / "data" / (
                name if name == "supervision_v1.json"
                else "recalibration_v2_r1/" + name)).read_bytes()
            for name in FILE_PINS
        }
        self.worker_raw = (ROOT / "validation_task_worker.py").read_bytes()
        self.backend_raw = (ROOT / "validation_remote.py").read_bytes()
        self.source_raw = {
            name: (ROOT / name).read_bytes()
            for name in (
                "pilot_protocol.py", "validation_contract.py", "validation_report.py",
                "validation_real_report.py", "recalibration_data_contract.py",
                "recalibration_validation.py",
            )
        }
        self.stage_id = "f" * 32
        self.run_ids = ["%032x" % ordinal for ordinal in range(15)]

    def prepare(self):
        return package.build_preparation(
            self.raw_files, self.worker_raw, self.backend_raw, self.source_raw,
            stage_id=self.stage_id, run_ids=self.run_ids)

    def test_first_preparation_is_complete_but_not_executable(self):
        prepared = self.prepare()
        self.assertEqual(prepared["public"]["status"], "PREPARATION_ONLY")
        self.assertEqual(len(prepared["private"]["bundles_hex"]), 15)
        self.assertIs(prepared["public"]["launch_ready"], False)
        self.assertEqual(prepared["public"]["counts"],
                         {"scheduled": 480, "batches": 15, "references": 192,
                          "witnesses": 36})
```

Before final GREEN, expand tests to cover each public contract (write/run failures before implementation changes):

1. Exact public/private field sets, digest reconstruction and all false flags/missing bindings.
2. Independent 480-request projection built from raw JSON prompts/oracles, not by calling `build_batches` for the expected result; verify all 15 exact bundle bytes, ordinal/request IDs and boundary 31/32/39/40.
3. Stable equality and canonical encoded bytes across repeat calls and reversed input-mapping insertion order.
4. Every frozen data, worker, backend and dependency-source pin rejects appended bytes.
5. Source mapping missing/extra/wrong keys, dict/key/value subclasses, bytearray/string values, and over-limit buffers fail closed.
6. Stage malformed/uppercase/newline/wrong type/subclass and stage-batch collision; run IDs wrong length/container/type/duplicate/case all rejected.
7. Separate preparations with changed valid stage or batch IDs change the appropriate manifest/bundle binding; same raw inputs are not mutated.
8. Validator accepts the reconstructed preparation and returns None, but rejects changed public values, rehashed forged metadata/flags, swapped/edited/uppercased private hex, missing/extra fields and bool/int substitution.
9. Validator rejects a preparation from other stage/run IDs or other raw inputs even after its public digest is recomputed.
10. Returned mutable containers do not alias caller inputs, constants or subsequent preparations.
11. No imports of backend/worker/tokenizer/model or production I/O/process/network operations. Use AST import/call inspection and interpreter module observations; don't launch SSH or execute a stored-source string to prove non-execution.
12. Six-file data/source subset coverage stays explicit; no claims, commands, preflight, acceptance or complete-source-manifest fields can be added by a caller.

- [x] **Step 2: Preserve actual RED**

Run from repository root, saving raw stdout/stderr and the immediate exit code under the stage evidence root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_execution_package -v
```

Expected: assertion failures for missing module/behavior, no accidental import failure.
Do not proceed to production implementation until RED is recorded.

- [x] **Step 3: Implement the minimal pure module**

Candidate implementation, with no new imports beyond pure helpers:

```python
import hashlib
import re

from pilot_protocol import FLAGS, fields, require
from recalibration_data_contract import FILE_PINS, _plain
from recalibration_validation import WORKER_SHA256, build_batches
from validation_contract import digest, encoded


BACKEND_SHA256 = "9be98748dfd2bdcdd132c6ed4c04c3dd67f11d49d6049fcbe7960bde622690da"
SOURCE_PINS = {
    "pilot_protocol.py": "5074980a2b5b9d8ec0a52288d7cb62cbc847688f094cd8ba189a7135ba315c97",
    "validation_contract.py": "a4d6de138fa4817b23fc6e28442785b0d68a6b8f7360fb110582752d3463299d",
    "validation_report.py": "2e6bfdd61b60a08afdac436e2a2d83cc0b6878914a8d1aeb3a2a9a1da2e28a1e",
    "validation_real_report.py": "55d7de49c81236e60712bf0f578a152f9807f758feaca99a09b15e1d4579909c",
    "recalibration_data_contract.py": "52695a9e2fb7e90b04f4fc7c95dcdc8b0310bbb0c269e6779890b9dff989ed02",
    "recalibration_validation.py": "a65fe7813b9ffd9db20dd232350737a528716d03583255fa907706be6f9076ad",
}
MISSING_BINDINGS = (
    "complete_execution_sources", "live_preflight", "launch_configuration", "one_time_scope",
)
FALSE_FLAGS = (
    *FLAGS, "launch_ready", "run_authorized", "dispatch_authorized",
    "execution_source_manifest_complete", "execution_provenance_verified",
    "trusted_collection_accepted", "dynamic_checks_passed", "full_dynamic_validation_complete",
)


def _pinned_bytes(raw, expected):
    require(type(raw) is bytes and len(raw) <= 1048576, "source bytes")
    require(hashlib.sha256(raw).hexdigest() == expected, "source identity")


def build_preparation(raw_files, worker_raw, backend_raw, source_raw, *, stage_id, run_ids):
    require(type(stage_id) is str and re.fullmatch("[0-9a-f]{32}", stage_id) is not None,
            "stage ID")
    fields(source_raw, SOURCE_PINS)
    require(all(type(name) is str for name in source_raw), "source key type")
    for name, expected in SOURCE_PINS.items():
        _pinned_bytes(source_raw[name], expected)
    _pinned_bytes(backend_raw, BACKEND_SHA256)
    batches = build_batches(raw_files, worker_raw, run_ids)
    require(stage_id not in run_ids, "stage and batch IDs must differ")
    manifest = {
        "kind": "LORA_RECALIBRATION_EXECUTION_PREPARATION", "version": 1,
        "data_version": "recalibration_v2_r1", "status": "PREPARATION_ONLY",
        "stage_id": stage_id, "run_ids": list(run_ids),
        "file_sha256": dict(FILE_PINS), "dependency_source_sha256": dict(SOURCE_PINS),
        "backend_sha256": BACKEND_SHA256, "worker_sha256": WORKER_SHA256,
        "plan_sha256": batches[0]["plan_sha256"],
        "counts": {"scheduled": 480, "batches": 15, "references": 192, "witnesses": 36},
        "batches": [
            {**{name: value for name, value in batch.items() if name != "bundle_raw"},
             "bundle_bytes": len(batch["bundle_raw"])}
            for batch in batches
        ],
        "missing_bindings": list(MISSING_BINDINGS),
        **{flag: False for flag in FALSE_FLAGS},
    }
    return {
        "public": {**manifest, "preparation_sha256": digest(manifest)},
        "private": {"bundles_hex": [batch["bundle_raw"].hex() for batch in batches]},
    }


def validate_preparation(preparation, raw_files, worker_raw, backend_raw, source_raw, *,
                         stage_id, run_ids):
    _plain(preparation)
    expected = build_preparation(
        raw_files, worker_raw, backend_raw, source_raw, stage_id=stage_id, run_ids=run_ids)
    require(encoded(preparation) == encoded(expected), "preparation reconstruction differs")
```

The new module does not attest its own loaded source or claim a future controller exists.
Product code adjustments needed by a demonstrated test failure are allowed within the
interface and constraints; document the actual RED/GREEN evidence rather than treating this candidate as infallible.

- [x] **Step 4: Focused GREEN, then one narrow regression**

Focused command is the same as Step 2. Narrow regression:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_data_contract test_recalibration_validation test_recalibration_execution_package -v
```

Expected: all discovered tests pass with no skips/errors; regression count includes
the existing 21+27 plus the new tests, not focused-count addition.
Do not repeat the unrelated 157-test tokenizer suite.
Preserve final log/exit files, implementation/test hashes and preservation check.

- [x] **Step 5: Report, task review and subplan closeout**

No commit: user constraints override the workflow template.
Write the full implementation/TDD report to the assigned private stage path.
Parent creates an actual diff of these two added files; a fresh reviewer reads
the task, report and diff for spec compliance and code quality.
Fix Critical/Important findings with focused regression and re-review.
Parent separately checks all 480 request identities from raw JSON and the preserved
baseline, then records result boundaries in the stage note/ledger.

Completed on 2026-11-20. Saved RED: 18 assertion failures; focused GREEN: 18
passed; narrow regression: 66 passed, including the focused tests. Parent
independently verified 480 request identities, 15 bundle identities and all 184
baseline files. Fresh reviewer Singer found no P0/P1/P2 issues against unchanged
source/test/diff hashes. No code correction or unchanged-test rerun was needed.
Result note:
`experiments/lora_necessity/20261120-recalibration-execution-preparation-results.md`.
Evidence root: `/private/tmp/lora-recalibration-package-20261120-2GDyzS`.
This completes only the preparation subplan, not live admission or execution.

## Plan Self-Review

- Covered here: fixed-data/source subset identity, exact batch preparation, stage
  separation, private payload/public digest boundary, deterministic reconstruction,
  explicit non-executable state and mutation/admission rejection.
- Not covered here: full source closure including this new module and future
  controller, runtime/preflight binding, exact SSH/paths/claims, framing, deadlines,
  real supervisor/controller/evidence acceptance. These require subsequent subplans;
  no current output claims those prerequisites complete.
- This narrow subplan has one independently testable deliverable. Do not expand it
  into the other three proposed modules or a real-run configuration.
- Function signatures and raw byte parameter order are identical in the candidate,
  test scaffold and contract; changed prepared flags must fail full reconstruction.
- Existing checkout/stdlib testing/no publication constraints supersede workflow
  templates requiring worktree creation, installation, commits or test telemetry.
