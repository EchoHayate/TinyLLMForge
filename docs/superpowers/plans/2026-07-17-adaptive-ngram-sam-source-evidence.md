# Adaptive N-Gram SAM Source Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the adaptive n-gram remote gate produce self-contained,
reconstructable evidence for the exact uncommitted K1 fast-path source that
ran remotely, including a same-snapshot remote K1 regression test.

**Architecture:** Add standard-library source snapshot, hashing, patch
reconstruction, and artifact verification helpers to
`tools/adaptive_ngram_gate.py`. Embed the source identity and remote preflight
record into the gate manifest and rows, then change the remote runner to stage
one immutable source tree, verify it remotely, run the K1 test, and archive
that same tree with the model artifacts. The local verifier rebuilds the
owned source from the recorded base commit plus `source.patch` and rejects any
identity mismatch before trusting the performance decision.

**Tech Stack:** Python 3 standard library, Git CLI, dependency-light Python
test scripts, Bash, SSH/SCP, existing TinyLLMForge adaptive n-gram gate,
Qwen3-0.6B on the remote CUDA host.

## Global Constraints

- The normative design is
  `docs/superpowers/specs/2026-07-17-adaptive-ngram-sam-source-evidence-design.md`.
- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`; do not modify
  `/Users/bytedance/dev/TinyLLMForge`.
- Execute inline in the current session; do not dispatch subagents.
- Keep `tools/profile_ngram_commit.py` and
  `tools/test_ngram_speculative.py` uncommitted until the remote canonical K1
  gate passes correctness, source reconstruction, and performance thresholds.
- This plan changes evidence and orchestration only. Do not change adaptive
  policy, prompt bank, thresholds, K1 behavior, model code, or runtime
  scheduling.
- The owned source boundary is every regular file under `tinyvllm/` plus
  `tools/draft_model_schema.py`, `tools/profile_ngram_commit.py`,
  `tools/adaptive_ngram_gate.py`, `tools/test_ngram_speculative.py`,
  `tools/test_adaptive_ngram_gate.py`, and
  `tools/run_adaptive_ngram_gate_remote.sh`.
- Reject staged, unstaged, or untracked paths outside the owned boundary
  before a canonical snapshot. Reject untracked files inside the owned
  boundary.
- The exact staged bytes are the only bytes uploaded, executed, and archived;
  never read the mutable worktree a second time for upload.
- `source.patch` is generated against the full 40-character base commit with
  `git diff --binary --no-ext-diff`.
- Evidence failures classify as `INCOMPLETE`, not semantic or performance
  `NO_GO`.
- Remote model work runs only on `sitian@10.232.195.203` with Python
  `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python` and model
  `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`.
- Every model process continues to receive distinct dynamic
  `TINYVLLM_DIST_PORT` and `MASTER_PORT` values.
- Use a run-local temporary directory and shell trap. Do not clear shared
  `/tmp`, kill other users' processes, or modify the remote source checkout.
- Commit source-evidence implementation in selective commits that exclude the
  two K1 dirty files.
- Do not claim K1 performance improvement until the remote canonical artifact
  passes the full source and result verifier.

## File Structure

- Modify `tools/adaptive_ngram_gate.py`: owned-path expansion, source hashing,
  snapshot/evidence creation, patch reconstruction, preflight validation,
  manifest integration, artifact source materialization, row identity, resume
  identity, and full downloaded verification.
- Modify `tools/test_adaptive_ngram_gate.py`: temporary-Git source fixtures,
  source-evidence unit tests, manifest/resume tests, preflight failure tests,
  row identity tests, and artifact tamper tests.
- Modify `tools/run_adaptive_ngram_gate_remote.sh`: local immutable staging,
  remote source verification, remote K1 test recording, evidence-aware run
  invocation, and self-contained artifact download.
- Preserve `tools/profile_ngram_commit.py`: existing uncommitted K1 candidate;
  no evidence-task edits.
- Preserve the current uncommitted behavior change in
  `tools/test_ngram_speculative.py`; the runner uploads and executes it, but
  evidence-task commits do not include it.
- Modify `AGENT_HANDOFF_STATE.md` only after real remote evidence exists.
- Generate `experiments/adaptive_ngram/${RUN_TAG}/source/`,
  `source_evidence.json`, `source.patch`, and `source_preflight.json` only
  through the runner.

---

### Task 1: Deterministic Source Evidence Core

**Files:**
- Modify: `tools/test_adaptive_ngram_gate.py`
- Modify: `tools/adaptive_ngram_gate.py`

**Interfaces:**
- Produces:
  `OWNED_SOURCE_ROOTS: tuple[str, ...]`
- Produces:
  `expand_owned_source_paths(repo_root: Path) -> tuple[str, ...]`
- Produces:
  `hash_source_tree(source_root: Path, relative_paths: tuple[str, ...]) -> list[dict]`
- Produces:
  `source_tree_sha256(files: list[dict]) -> str`
- Produces:
  `build_source_evidence(repo_root: Path, out_dir: Path) -> dict`
- Produces:
  `validate_source_snapshot(source_root: Path, evidence: dict, patch_path: Path) -> dict`
- Produces:
  `reconstruct_source_snapshot(repo_root: Path, source_root: Path, evidence: dict, patch_path: Path) -> None`

- [ ] **Step 1: Add temporary-Git fixture helpers and failing source tests**

Add these imports to `tools/test_adaptive_ngram_gate.py`:

```python
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
```

Add a minimal fixture that creates only the owned paths needed by the test:

```python
def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


def _source_repo() -> tuple[tempfile.TemporaryDirectory, Path]:
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    (root / "tinyvllm").mkdir()
    (root / "tinyvllm" / "__init__.py").write_text("VALUE = 1\n")
    (root / "tools").mkdir()
    for name in (
        "draft_model_schema.py",
        "profile_ngram_commit.py",
        "adaptive_ngram_gate.py",
        "test_ngram_speculative.py",
        "test_adaptive_ngram_gate.py",
        "run_adaptive_ngram_gate_remote.sh",
    ):
        (root / "tools" / name).write_text(f"# {name}\n")
    _run(["git", "init"], root)
    _run(["git", "config", "user.name", "Gate Test"], root)
    _run(["git", "config", "user.email", "gate@example.invalid"], root)
    _run(["git", "add", "."], root)
    _run(["git", "commit", "-m", "base"], root)
    return temporary, root
```

Add tests with exact assertions:

```python
def test_source_evidence_reconstructs_dirty_owned_files():
    temporary, root = _source_repo()
    try:
        (root / "tools" / "profile_ngram_commit.py").write_text(
            "# profile_ngram_commit.py\nFAST_K1 = True\n"
        )
        out_dir = root / "snapshot"
        evidence = gate.build_source_evidence(root, out_dir)

        assert evidence["schema_version"] == 1
        assert evidence["dirty"] is True
        assert evidence["base_commit"] == _run(
            ["git", "rev-parse", "HEAD"], root
        ).stdout.strip()
        assert evidence["patch_size_bytes"] > 0
        assert evidence["tree_sha256"] == gate.source_tree_sha256(
            evidence["files"]
        )

        reconstructed = root / "reconstructed"
        gate.reconstruct_source_snapshot(
            root,
            reconstructed,
            evidence,
            out_dir / "source.patch",
        )
        gate.validate_source_snapshot(
            reconstructed,
            evidence,
            out_dir / "source.patch",
        )
        assert (
            reconstructed / "tools" / "profile_ngram_commit.py"
        ).read_text().endswith("FAST_K1 = True\n")
    finally:
        temporary.cleanup()


def test_source_evidence_clean_tree_uses_empty_patch():
    temporary, root = _source_repo()
    try:
        out_dir = root / "snapshot"
        evidence = gate.build_source_evidence(root, out_dir)
        assert evidence["dirty"] is False
        assert (out_dir / "source.patch").read_bytes() == b""
        assert evidence["patch_sha256"] == gate.sha256_bytes(b"")
    finally:
        temporary.cleanup()


def test_source_evidence_rejects_untracked_owned_file():
    temporary, root = _source_repo()
    try:
        (root / "tinyvllm" / "untracked.py").write_text("unexpected = True\n")
        try:
            gate.build_source_evidence(root, root / "snapshot")
        except ValueError as exc:
            assert "untracked owned source" in str(exc)
        else:
            raise AssertionError("untracked owned source must fail")
    finally:
        temporary.cleanup()


def test_validate_source_snapshot_rejects_changed_missing_and_extra_files():
    temporary, root = _source_repo()
    try:
        out_dir = root / "snapshot"
        evidence = gate.build_source_evidence(root, out_dir)
        source_root = out_dir / "source"

        target = source_root / "tinyvllm" / "__init__.py"
        original = target.read_bytes()
        target.write_bytes(b"changed\n")
        for expected in ("source file hash mismatch",):
            try:
                gate.validate_source_snapshot(
                    source_root, evidence, out_dir / "source.patch"
                )
            except ValueError as exc:
                assert expected in str(exc)
            else:
                raise AssertionError(expected)
        target.write_bytes(original)

        target.unlink()
        try:
            gate.validate_source_snapshot(
                source_root, evidence, out_dir / "source.patch"
            )
        except ValueError as exc:
            assert "source path set mismatch" in str(exc)
        else:
            raise AssertionError("missing source path must fail")
        target.write_bytes(original)

        extra = source_root / "tinyvllm" / "extra.py"
        extra.write_text("extra = True\n")
        try:
            gate.validate_source_snapshot(
                source_root, evidence, out_dir / "source.patch"
            )
        except ValueError as exc:
            assert "source path set mismatch" in str(exc)
        else:
            raise AssertionError("extra source path must fail")
    finally:
        temporary.cleanup()


def test_validate_source_snapshot_rejects_patch_and_tree_tampering():
    temporary, root = _source_repo()
    try:
        out_dir = root / "snapshot"
        evidence = gate.build_source_evidence(root, out_dir)
        patch = out_dir / "source.patch"
        patch.write_bytes(patch.read_bytes() + b"x")
        try:
            gate.validate_source_snapshot(out_dir / "source", evidence, patch)
        except ValueError as exc:
            assert "patch hash mismatch" in str(exc)
        else:
            raise AssertionError("changed patch must fail")

        patch.write_bytes(b"")
        changed = dict(evidence)
        changed["tree_sha256"] = "0" * 64
        try:
            gate.validate_source_snapshot(out_dir / "source", changed, patch)
        except ValueError as exc:
            assert "source tree hash mismatch" in str(exc)
        else:
            raise AssertionError("changed tree hash must fail")
    finally:
        temporary.cleanup()
```

Append every new test to `main()`.

- [ ] **Step 2: Run the source tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_adaptive_ngram_gate.py
```

Expected: fail because `build_source_evidence` or another new helper does not
exist. The failure must be an assertion/import failure caused by the missing
source-evidence behavior, not a fixture error.

- [ ] **Step 3: Implement hashing, snapshot, validation, and reconstruction**

In `tools/adaptive_ngram_gate.py`, add imports:

```python
import shutil
import tempfile
```

Add constants:

```python
OWNED_SOURCE_ROOTS = (
    "tinyvllm",
    "tools/draft_model_schema.py",
    "tools/profile_ngram_commit.py",
    "tools/adaptive_ngram_gate.py",
    "tools/test_ngram_speculative.py",
    "tools/test_adaptive_ngram_gate.py",
    "tools/run_adaptive_ngram_gate_remote.sh",
)
```

Implement the interfaces with these exact invariants:

```python
def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git(repo_root: Path, *args: str, input_bytes: bytes | None = None):
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        input=input_bytes,
        capture_output=True,
        check=False,
    )


def source_tree_sha256(files: list[dict]) -> str:
    canonical = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256_bytes(canonical)
```

`expand_owned_source_paths()` must:

- resolve each owned root under `repo_root`;
- reject missing paths and symlinks;
- recursively include only regular files for directories;
- use POSIX repository-relative paths;
- return one sorted tuple without duplicates.

`hash_source_tree()` must return sorted records of exactly:

```python
{
    "path": relative_path,
    "size_bytes": len(payload),
    "sha256": sha256_bytes(payload),
}
```

`build_source_evidence()` must:

- require `repo_root/.git` to resolve through Git;
- resolve `base_commit` with `git rev-parse HEAD`;
- reject untracked paths returned by
  `git status --porcelain=v1 --untracked-files=all` when they are inside an
  owned root;
- reject any changed path outside the owned roots;
- create `out_dir/source/`;
- copy every expanded owned path with `shutil.copyfile`;
- generate binary patch bytes using:

```python
patch_result = _git(
    repo_root,
    "diff",
    "--binary",
    "--no-ext-diff",
    base_commit,
    "--",
    *OWNED_SOURCE_ROOTS,
)
```

- write `out_dir/source.patch`;
- build the schema from the design;
- write `out_dir/source_evidence.json`;
- call `reconstruct_source_snapshot()` in a temporary directory;
- call `validate_source_snapshot()` on both staged and reconstructed trees.

`validate_source_snapshot()` must reject:

- wrong schema or malformed SHA values;
- patch size/hash mismatch;
- any source path-set mismatch;
- any file size/hash mismatch;
- any recomputed tree hash mismatch.

`reconstruct_source_snapshot()` must:

- use `git archive <base_commit> <owned roots>` piped to `tar -xf -` in a
  temporary destination;
- apply non-empty patch bytes with
  `git apply --binary --unsafe-paths --directory=<destination>`;
- validate the destination against the evidence.

- [ ] **Step 4: Run source tests and existing gate tests GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_adaptive_ngram_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  tools/adaptive_ngram_gate.py \
  tools/test_adaptive_ngram_gate.py
git diff --check
```

Expected:

- `adaptive ngram gate tests passed`;
- compilation succeeds;
- no whitespace errors.

- [ ] **Step 5: Commit only the source-evidence core**

Run:

```bash
git add tools/adaptive_ngram_gate.py tools/test_adaptive_ngram_gate.py
git diff --cached --name-only
git commit -m "test: add adaptive gate source evidence"
```

Expected staged paths are exactly the two named files. Do not stage
`tools/profile_ngram_commit.py` or `tools/test_ngram_speculative.py`.

---

### Task 2: Manifest, Resume, Rows, and Artifact Verification

**Files:**
- Modify: `tools/test_adaptive_ngram_gate.py`
- Modify: `tools/adaptive_ngram_gate.py`

**Interfaces:**
- Modifies:
  `build_manifest(..., source_evidence: dict, source_preflight: dict) -> dict`
- Modifies:
  `run_gate(..., source_evidence_path: Path, source_patch_path: Path, source_preflight_path: Path) -> dict`
- Produces:
  `validate_source_preflight(preflight: dict, evidence: dict) -> None`
- Produces:
  `materialize_source_artifacts(out_dir: Path, source_root: Path, evidence_path: Path, patch_path: Path, preflight_path: Path) -> None`
- Modifies:
  `verify_artifacts(out_dir: Path, repo_root: Path = _REPO_ROOT) -> dict`

- [ ] **Step 1: Add failing manifest and artifact tests**

Add a valid synthetic preflight helper:

```python
def _source_preflight(evidence: dict) -> dict:
    return {
        "schema_version": 1,
        "source_tree_sha256": evidence["tree_sha256"],
        "source_verify": {
            "returncode": 0,
            "stdout_sha256": gate.sha256_text("source verified\n"),
            "stderr_sha256": gate.sha256_text(""),
        },
        "k1_test": {
            "command": [
                "python3",
                "tools/test_ngram_speculative.py",
            ],
            "returncode": 0,
            "stdout_sha256": gate.sha256_text(
                "ngram speculative tests passed\n"
            ),
            "stderr_sha256": gate.sha256_text(""),
        },
    }
```

Add focused tests:

```python
def test_manifest_embeds_source_identity_and_rows_copy_it():
    temporary, root = _source_repo()
    try:
        snapshot = root / "snapshot"
        evidence = gate.build_source_evidence(root, snapshot)
        preflight = _source_preflight(evidence)
        manifest = gate.build_manifest(
            repetitions=1,
            base_seed=20260714,
            source_commit=evidence["base_commit"],
            source_dirty=evidence["dirty"],
            model_path="/models/Qwen3-0.6B",
            model_identifier="Qwen3-0.6B",
            host="synthetic-host",
            python_bin="python3",
            source_evidence=evidence,
            source_preflight=preflight,
        )
        assert manifest["schema_version"] == 2
        assert manifest["source_tree_sha256"] == evidence["tree_sha256"]
        assert manifest["source_evidence"] == evidence
        assert manifest["source_preflight"] == preflight

        spec = manifest["run_specs"][0]
        row, _ = gate._normalize_row(
            manifest,
            spec,
            {"summary": {}, "per_prompt": []},
            {
                "returncode": 1,
                "command": [],
                "tinyvllm_dist_port": 20000,
                "master_port": 20001,
            },
        )
        assert row["source_tree_sha256"] == evidence["tree_sha256"]
    finally:
        temporary.cleanup()


def test_source_preflight_must_match_and_pass():
    temporary, root = _source_repo()
    try:
        evidence = gate.build_source_evidence(root, root / "snapshot")
        gate.validate_source_preflight(_source_preflight(evidence), evidence)

        for mutation, expected in (
            (lambda value: value["k1_test"].update(returncode=1),
             "remote K1 test failed"),
            (lambda value: value.update(source_tree_sha256="0" * 64),
             "preflight source tree mismatch"),
        ):
            value = json.loads(json.dumps(_source_preflight(evidence)))
            mutation(value)
            try:
                gate.validate_source_preflight(value, evidence)
            except ValueError as exc:
                assert expected in str(exc)
            else:
                raise AssertionError(expected)
    finally:
        temporary.cleanup()


def test_structural_failures_reject_row_source_identity_mismatch():
    manifest, rows, events = _synthetic_complete_gate_rows(repetitions=1)
    rows[0]["source_tree_sha256"] = "0" * 64
    summary = gate.summarize_rows(manifest, rows, events)
    assert summary["decision"] == "INCOMPLETE"
    assert any(
        "source_tree_sha256_mismatch" in item
        for item in summary["structural_failures"]
    )
```

Add a self-contained artifact fixture that writes manifest, rows, events,
summary, report, source tree, evidence, patch, and preflight, then add:

```python
def test_verify_artifacts_reconstructs_recorded_source():
    temporary, root, out_dir = _complete_artifact_fixture()
    try:
        summary = gate.verify_artifacts(out_dir, repo_root=root)
        assert summary["decision"] == "GO"
    finally:
        temporary.cleanup()


def test_verify_artifacts_rejects_source_patch_and_preflight_tampering():
    for relative_path, mutate, expected in (
        (
            "source/tools/profile_ngram_commit.py",
            lambda path: path.write_text(path.read_text() + "tamper\n"),
            "source file hash mismatch",
        ),
        (
            "source.patch",
            lambda path: path.write_bytes(path.read_bytes() + b"x"),
            "patch hash mismatch",
        ),
        (
            "source_preflight.json",
            lambda path: path.write_text(
                json.dumps({
                    **json.loads(path.read_text()),
                    "k1_test": {
                        **json.loads(path.read_text())["k1_test"],
                        "returncode": 1,
                    },
                })
            ),
            "remote K1 test failed",
        ),
    ):
        temporary, root, out_dir = _complete_artifact_fixture()
        try:
            mutate(out_dir / relative_path)
            try:
                gate.verify_artifacts(out_dir, repo_root=root)
            except ValueError as exc:
                assert expected in str(exc)
            else:
                raise AssertionError(expected)
        finally:
            temporary.cleanup()
```

Update synthetic manifest/row fixtures to carry valid source evidence,
preflight, and `source_tree_sha256`. Append new tests to `main()`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_adaptive_ngram_gate.py
```

Expected: fail because manifest/preflight/artifact integration is absent.

- [ ] **Step 3: Integrate source identity into manifest and rows**

Change `build_manifest()` to require `source_evidence` and
`source_preflight`, call both validators, and return:

```python
{
    "schema_version": 2,
    ...
    "source_commit": source_evidence["base_commit"],
    "source_dirty": source_evidence["dirty"],
    "source_tree_sha256": source_evidence["tree_sha256"],
    "source_evidence": source_evidence,
    "source_preflight": source_preflight,
}
```

Reject caller-provided `source_commit` or `source_dirty` when they disagree
with the evidence. Add `source_tree_sha256` to every normalized row.

In `_structural_failures()`, require every row's:

- `source_commit`;
- `source_dirty`;
- `source_tree_sha256`

to equal the manifest.

- [ ] **Step 4: Add source artifact materialization and full verification**

`materialize_source_artifacts()` must copy:

- staged source root to `out_dir/source/`;
- evidence to `out_dir/source_evidence.json`;
- patch to `out_dir/source.patch`;
- preflight to `out_dir/source_preflight.json`.

It must validate before and after copying.

`run_gate()` must:

- load and validate the three source inputs before writing `manifest.json`;
- call `materialize_source_artifacts()` before starting model processes;
- include source identity fields in resume comparison;
- reject resume when any evidence/preflight object differs.

`verify_artifacts()` must:

- load source evidence, patch, preflight, and source tree from `out_dir`;
- require exact equality with manifest embedded objects;
- validate the source tree and preflight;
- reconstruct base commit plus patch in a temporary directory;
- compare the reconstruction to the artifact evidence;
- then perform existing row/summary/report recomputation.

Add CLI run arguments:

```python
run_parser.add_argument("--source-root", type=Path, required=True)
run_parser.add_argument("--source-evidence", type=Path, required=True)
run_parser.add_argument("--source-patch", type=Path, required=True)
run_parser.add_argument("--source-preflight", type=Path, required=True)
```

Add commands:

```python
snapshot_parser = subparsers.add_parser("snapshot-source")
snapshot_parser.add_argument("--repo-root", type=Path, required=True)
snapshot_parser.add_argument("--out-dir", type=Path, required=True)

source_verify_parser = subparsers.add_parser("verify-source")
source_verify_parser.add_argument("--source-root", type=Path, required=True)
source_verify_parser.add_argument("--evidence", type=Path, required=True)
source_verify_parser.add_argument("--patch", type=Path, required=True)
```

`snapshot-source` prints the evidence JSON. `verify-source` prints:

```json
{
  "valid": true,
  "source_tree_sha256": "<tree hash>"
}
```

- [ ] **Step 5: Run complete dependency-light gate tests GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_adaptive_ngram_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/adaptive_ngram_gate.py --help
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  tools/adaptive_ngram_gate.py \
  tools/test_adaptive_ngram_gate.py
git diff --check
```

Expected all commands succeed.

- [ ] **Step 6: Commit manifest and verifier integration**

Run:

```bash
git add tools/adaptive_ngram_gate.py tools/test_adaptive_ngram_gate.py
git diff --cached --name-only
git commit -m "feat: verify adaptive gate source artifacts"
```

Expected staged paths are exactly the two named files.

---

### Task 3: Immutable Remote Staging and K1 Preflight

**Files:**
- Modify: `tools/test_adaptive_ngram_gate.py`
- Modify: `tools/adaptive_ngram_gate.py`
- Modify: `tools/run_adaptive_ngram_gate_remote.sh`

**Interfaces:**
- Produces:
  `write_source_preflight(source_root: Path, evidence_path: Path, patch_path: Path, command_record_path: Path, out_path: Path) -> dict`
- Runner produces staged:
  `${STAGING_DIR}/{source,source_evidence.json,source.patch}`
- Runner produces remote:
  `${REMOTE_DIR}/source_preflight.json`
- Runner invokes gate with all four source arguments.

- [ ] **Step 1: Add failing structured preflight-record tests**

Add a test for converting recorded command JSON into a validated preflight:

```python
def test_write_source_preflight_records_verified_tree_and_k1_test():
    temporary, root = _source_repo()
    try:
        snapshot = root / "snapshot"
        evidence = gate.build_source_evidence(root, snapshot)
        command_record = root / "commands.json"
        command_record.write_text(json.dumps({
            "source_verify": {
                "returncode": 0,
                "stdout": "source verified\n",
                "stderr": "",
            },
            "k1_test": {
                "command": [
                    "python3",
                    "tools/test_ngram_speculative.py",
                ],
                "returncode": 0,
                "stdout": "ngram speculative tests passed\n",
                "stderr": "",
            },
        }))
        output = root / "source_preflight.json"
        preflight = gate.write_source_preflight(
            snapshot / "source",
            snapshot / "source_evidence.json",
            snapshot / "source.patch",
            command_record,
            output,
        )
        gate.validate_source_preflight(preflight, evidence)
        assert json.loads(output.read_text()) == preflight
        assert preflight["k1_test"]["stdout_sha256"] == gate.sha256_text(
            "ngram speculative tests passed\n"
        )
    finally:
        temporary.cleanup()
```

Add failure variants for a nonzero source verification or K1 return code.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_adaptive_ngram_gate.py
```

Expected: missing `write_source_preflight`.

- [ ] **Step 3: Implement structured preflight writer and CLI**

Implement `write_source_preflight()` to:

- validate the source snapshot first;
- load a command record containing raw stdout/stderr only in the remote
  temporary area;
- emit schema version 1;
- include source tree identity;
- preserve the K1 command;
- preserve return codes;
- replace raw outputs with SHA-256 fields in the canonical record;
- validate the final preflight;
- atomically write JSON.

Add:

```python
preflight_parser = subparsers.add_parser("write-source-preflight")
preflight_parser.add_argument("--source-root", type=Path, required=True)
preflight_parser.add_argument("--evidence", type=Path, required=True)
preflight_parser.add_argument("--patch", type=Path, required=True)
preflight_parser.add_argument("--command-record", type=Path, required=True)
preflight_parser.add_argument("--out", type=Path, required=True)
```

- [ ] **Step 4: Replace mutable worktree upload with one staged snapshot**

In `tools/run_adaptive_ngram_gate_remote.sh`:

1. Keep current host, model discovery, SSH ControlMaster, mode, run tag, and
   repetition behavior.
2. Replace broad `SOURCE_DIRTY` discovery and direct tar from `REPO_ROOT` with:

```bash
STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/adaptive-ngram-sam.XXXXXX")"
cleanup() {
  rm -rf "${STAGING_DIR}"
}
trap cleanup EXIT

PYTHONDONTWRITEBYTECODE=1 python3 \
  "${REPO_ROOT}/tools/adaptive_ngram_gate.py" \
  snapshot-source \
  --repo-root "${REPO_ROOT}" \
  --out-dir "${STAGING_DIR}"
```

3. Read `base_commit`, `dirty`, and `tree_sha256` from
   `${STAGING_DIR}/source_evidence.json` using Python, not `jq`.
4. Create `${REMOTE_DIR}` and tar `${STAGING_DIR}/source/.` into it.
5. Copy `source_evidence.json` and `source.patch` into `${REMOTE_DIR}`.
6. Never tar files directly from `${REPO_ROOT}` after the snapshot command.

- [ ] **Step 5: Add remote source verification and K1 test before GPU work**

Run these commands in `${REMOTE_DIR}`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${REMOTE_DIR}" \
  "${REMOTE_PYTHON}" tools/adaptive_ngram_gate.py verify-source \
  --source-root "${REMOTE_DIR}" \
  --evidence "${REMOTE_DIR}/source_evidence.json" \
  --patch "${REMOTE_DIR}/source.patch"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${REMOTE_DIR}" \
  "${REMOTE_PYTHON}" tools/test_ngram_speculative.py
```

Capture each return code, stdout, and stderr into run-local files without
allowing `set -e` to skip record generation. Write `command_record.json` using
the remote Python standard library, then invoke:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${REMOTE_DIR}" \
  "${REMOTE_PYTHON}" tools/adaptive_ngram_gate.py write-source-preflight \
  --source-root "${REMOTE_DIR}" \
  --evidence "${REMOTE_DIR}/source_evidence.json" \
  --patch "${REMOTE_DIR}/source.patch" \
  --command-record "${REMOTE_DIR}/command_record.json" \
  --out "${REMOTE_DIR}/source_preflight.json"
```

If either command failed, `write-source-preflight` must fail and no model run
starts.

- [ ] **Step 6: Pass source evidence into the gate and download it**

Extend remote `RUN_ARGS` with:

```bash
--source-root "${REMOTE_DIR}"
--source-evidence "${REMOTE_DIR}/source_evidence.json"
--source-patch "${REMOTE_DIR}/source.patch"
--source-preflight "${REMOTE_DIR}/source_preflight.json"
```

Remove standalone `--source-commit` and `--source-dirty` arguments after the
Python CLI no longer requires them.

Keep the existing local `verify --out-dir "${LOCAL_OUT}"`. It now verifies
source reconstruction as well as rows/summary/report.

- [ ] **Step 7: Run shell and dependency-light tests GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_adaptive_ngram_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
bash -n tools/run_adaptive_ngram_gate_remote.sh
git diff --check
```

Expected:

- adaptive gate tests pass;
- K1 regression test passes locally;
- shell syntax passes;
- no whitespace errors.

- [ ] **Step 8: Commit only evidence runner changes**

Run:

```bash
git add \
  tools/adaptive_ngram_gate.py \
  tools/test_adaptive_ngram_gate.py \
  tools/run_adaptive_ngram_gate_remote.sh
git diff --cached --name-only
git commit -m "feat: stage auditable adaptive gate source"
```

Expected staged paths are exactly the three named files. The dirty K1 files
remain unstaged.

---

### Task 4: Local Completion Audit and Remote Preflight

**Files:**
- Inspect: `tools/profile_ngram_commit.py`
- Inspect: `tools/test_ngram_speculative.py`
- Inspect: `tools/adaptive_ngram_gate.py`
- Inspect: `tools/run_adaptive_ngram_gate_remote.sh`
- Inspect: `docs/superpowers/specs/2026-07-17-adaptive-ngram-sam-source-evidence-design.md`

**Interfaces:**
- Consumes all previous task outputs.
- Produces no source change unless a failing test exposes a defect.

- [ ] **Step 1: Verify the K1 dirty diff is still exactly the approved behavior**

Run:

```bash
git diff -- tools/profile_ngram_commit.py tools/test_ngram_speculative.py
```

Require:

- no unconditional sync after first-target decode;
- prepare sync only when `query_len > 0`;
- tail sync only when `query_len > 0`;
- K1 test asserts `[4]`, `query_len == 0`, zero spec-verify calls, no prepare
  calls, and no sync calls;
- no unrelated K1 source changes.

- [ ] **Step 2: Run focused and adjacent local regressions**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_adaptive_ngram_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_oracle.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_model_runner_spec_verify.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  tools/adaptive_ngram_gate.py \
  tools/profile_ngram_commit.py \
  tools/test_adaptive_ngram_gate.py \
  tools/test_ngram_speculative.py
bash -n tools/run_adaptive_ngram_gate_remote.sh
git diff --check
```

Expected every command succeeds.

- [ ] **Step 3: Perform prompt-to-artifact coverage audit**

Create an inline checklist and inspect concrete code/tests for:

1. exact owned-source paths;
2. local staging hashes;
3. binary patch and hash;
4. local reconstruction;
5. immutable upload;
6. remote rehash;
7. remote K1 test;
8. preflight manifest;
9. row source identity;
10. resume source identity;
11. archived source tree;
12. downloaded file/hash verification;
13. downloaded base-plus-patch reconstruction;
14. raw/summary/report recomputation;
15. K1 output equality and performance thresholds;
16. unique model-process ports.

Treat any missing direct evidence as incomplete and fix it with a new RED/GREEN
cycle before remote execution.

- [ ] **Step 4: Check remote authentication without starting GPU work**

Run:

```bash
ssh sitian@10.232.195.203 'printf remote-ok'
```

If the ControlMaster is required, use the existing
`/tmp/ssh-sitian-10.232.195.203` socket only when it exists. If authentication
fails due to expired Kerberos credentials, stop remote work and report the
exact error; do not modify or weaken SSH settings.

- [ ] **Step 5: Run evidence-aware remote preflight**

Run:

```bash
RUN_TAG=20260717-k1-sam-preflight \
MODEL_PATH=/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B \
tools/run_adaptive_ngram_gate_remote.sh preflight
```

Expected:

- local snapshot/reconstruction passes;
- remote snapshot hashes pass;
- remote K1 test passes;
- no model process is launched;
- remote preflight prints the source tree hash.

---

### Task 5: Remote Smoke, Canonical Gate, and K1 Decision

**Files:**
- Generate:
  `experiments/adaptive_ngram/20260717-k1-sam-smoke/`
- Generate:
  `experiments/adaptive_ngram/20260717-k1-sam-canonical/`
- Modify after evidence:
  `AGENT_HANDOFF_STATE.md`
- Conditionally commit:
  `tools/profile_ngram_commit.py`
  and `tools/test_ngram_speculative.py`

**Interfaces:**
- Consumes evidence-aware runner and current K1 dirty source.
- Produces a source-attributable `GO`, `NO_GO`, or `INCOMPLETE`.

- [ ] **Step 1: Run one-repetition remote smoke**

Run:

```bash
RUN_TAG=20260717-k1-sam-smoke \
MODEL_PATH=/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B \
tools/run_adaptive_ngram_gate_remote.sh smoke
```

Require:

- remote K1 test return code zero;
- source evidence and patch reconstruction pass;
- 20/20 rows;
- unique dynamic port pairs;
- all candidate outputs token-identical to baseline;
- local downloaded verifier passes.

If source evidence or infrastructure fails, classify `INCOMPLETE` and fix the
evidence path before canonical. If semantic output fails, classify `NO_GO` and
do not run canonical.

- [ ] **Step 2: Independently inspect smoke artifact identity**

Run a Python one-liner that prints and asserts:

```python
manifest["source_tree_sha256"]
manifest["source_evidence"]["patch_sha256"]
manifest["source_preflight"]["k1_test"]["returncode"] == 0
len(raw_rows) == 20
len({
    port
    for row in raw_rows
    for port in (
        row["process"]["tinyvllm_dist_port"],
        row["process"]["master_port"],
    )
}) == 40
all(
    row["source_tree_sha256"] == manifest["source_tree_sha256"]
    for row in raw_rows
)
```

Do not rely only on the verifier exit code.

- [ ] **Step 3: Run seven-repetition remote canonical**

Run only after smoke passes:

```bash
RUN_TAG=20260717-k1-sam-canonical \
MODEL_PATH=/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B \
tools/run_adaptive_ngram_gate_remote.sh canonical
```

Require:

- 140/140 unique rows;
- seven repetitions for every prompt/policy pair;
- 280 unique dynamic port values;
- remote K1 test zero;
- exact baseline equality;
- trajectory replay and adaptive exercise pass;
- source tree, patch, reconstruction, summary, and report verification pass.

- [ ] **Step 4: Audit canonical result rather than trusting `GO` alone**

Inspect:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/adaptive_ngram_gate.py \
  verify \
  --out-dir experiments/adaptive_ngram/20260717-k1-sam-canonical
```

Then independently check:

- policy counts are 28 each;
- run-key set equals manifest specs;
- source hashes equal smoke only if the K1 bytes did not change;
- baseline/K1 output IDs match per repetition and prompt;
- fixed K1 median throughput and relevant threshold result;
- no failed process, missing row, duplicate row, reused port, or failed
  preflight.

- [ ] **Step 5: Apply the commit gate**

If canonical is complete, exact, source-valid, and meets the precommitted
performance gate:

```bash
git add tools/profile_ngram_commit.py tools/test_ngram_speculative.py
git diff --cached --name-only
git commit -m "perf: remove redundant K1 verifier syncs"
```

If performance is `NO_GO`, leave the two files uncommitted and record the
negative result. If evidence/infrastructure is `INCOMPLETE`, leave them
uncommitted and record the blocker. Never commit based on smoke alone.

- [ ] **Step 6: Record exact evidence and limitations**

Update `AGENT_HANDOFF_STATE.md` with:

- K1 source tree hash, patch hash, and base commit;
- remote and local artifact paths;
- exact preflight, smoke, canonical, and verifier commands;
- row/repetition/port counts;
- exact correctness and performance decision;
- whether K1 files were committed;
- what the result proves and does not prove;
- next optimization direction if K1 is `NO_GO`.

Run:

```bash
git add AGENT_HANDOFF_STATE.md
git diff --cached --name-only
git commit -m "docs: record auditable K1 gate result"
```

Do not stage experiment artifacts unless repository policy already tracks that
artifact class.

---

## Final Verification

Before claiming completion, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_adaptive_ngram_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_oracle.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_model_runner_spec_verify.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
bash -n tools/run_adaptive_ngram_gate_remote.sh
git diff --check
git status --short
```

The final status must be explained path by path. A successful result must cite
the canonical artifact directory and full source identity. A blocked result
must retain the exact dirty K1 diff and name the concrete blocker.

