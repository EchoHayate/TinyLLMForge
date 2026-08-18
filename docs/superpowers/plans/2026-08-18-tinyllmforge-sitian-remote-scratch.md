# TinyLLMForge Sitian Remote Scratch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a safe, resumable workflow that stages the authoritative TinyLLMForge source and runs cache-producing CPU validation on `sitian:/data00/home/sitian` without creating TinyLLMForge cache, archive, or log trees on the macOS root data volume.

**Architecture:** Add one dependency-light Python CLI that owns fixed path policy, explicit-path validation, bounded SSH retries, streaming `git archive` initialization, explicit incremental tar sync, remote cache environment construction, bounded local status output, and remote receipts. Test its pure command construction and failure semantics with `unittest`, then run one real remote initialization and focused Task 5 CPU test while auditing both the remote workspace and local `/private/tmp`.

**Tech Stack:** Python 3 standard library, dataclasses, `argparse`, `subprocess`, `unittest`, Bash, OpenSSH, Kerberos file cache, bsdtar, GNU tar/sha256sum on sitian, Git.

## Global Constraints

- The authoritative date is Tuesday, August 18, 2026.
- Modify only `/Users/bytedance/Desktop/TinyLLMForge`, whose physical path is `/Users/bytedance/dev/TinyLLMForge`.
- Never read from, modify, stage, commit, package, or synchronize `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- The local checkout remains the only Git authority.
- The remote host is exactly `sitian@10.232.195.203`.
- The remote task root is exactly `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Do not modify the historical non-Git mirror `/data00/home/sitian/tllm/TinyLLMForge`.
- Do not create TinyLLMForge source archives, pycache trees, pytest caches, review logs, or validation logs under local `/`, `/tmp`, or `/private/tmp`.
- Any unavoidable local Python invocation sets `PYTHONDONTWRITEBYTECODE=1`, disables persistent test caches, produces bounded output, and does not redirect to a local log.
- All cache-producing tests and compile checks run on sitian with `TMPDIR`, `TMP`, `TEMP`, `PYTHONPYCACHEPREFIX`, and `XDG_CACHE_HOME` inside the remote task root.
- Initial source staging streams `git archive HEAD` directly to SSH; no local archive file is created.
- Incremental source staging accepts only explicit repository-relative paths.
- Never use `git add -A`, broad untracked-file discovery, or `rsync --delete`.
- Never synchronize `.git/`, `artifacts/`, `experiments/`, `__pycache__/`, `.pytest_cache/`, `*.pyc`, logs, PIDs, raw remote output, source archives, or `.superpowers/sdd/*review-package.diff`.
- Disable macOS xattrs and AppleDouble metadata during transport; the remote source must contain zero `._*` files.
- A failed transfer writes only to a unique remote staging directory and never deletes the last verified source.
- Do not refresh Kerberos credentials, kill SSH sessions, terminate remote processes, allocate GPUs, load models, or run a source-bound performance gate.
- Stage only explicit intended source, test, plan, audit, progress, and handoff files.
- Every commit uses `git -c core.hooksPath=/dev/null commit`.
- Every commit ends with exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Push every completed slice to `origin/feat/kv-sparse-attention`.

## File Map

- Create `tools/sitian_remote_scratch.py`: fixed configuration, path policy, SSH command construction, bounded retries, source initialization, explicit sync, remote command execution, and receipt summaries.
- Create `tools/test_sitian_remote_scratch.py`: standard-library unit tests for path rejection, command construction, retry behavior, staging promotion, remote environment, and bounded output.
- Create `docs/superpowers/audits/2026-08-18-tinyllmforge-sitian-remote-scratch-audit.md`: real remote initialization and no-local-cache acceptance evidence.
- Modify `.superpowers/sdd/progress.md`: record the storage-workflow interruption, remote scratch completion, and Task 5 resume point.
- Modify `AGENT_HANDOFF_STATE.md`: append the authoritative remote scratch paths, commands, verification result, and the unchanged GPU authorization boundary.

---

### Task 1: Fixed Path Policy and Safe Command Construction

**Files:**
- Create: `tools/test_sitian_remote_scratch.py`
- Create: `tools/sitian_remote_scratch.py`

**Interfaces:**
- Produces `ScratchConfig`.
- Produces `validate_relative_paths(paths: Sequence[str]) -> tuple[str, ...]`.
- Produces `remote_layout(config: ScratchConfig) -> dict[str, str]`.
- Produces `ssh_argv(config: ScratchConfig) -> tuple[str, ...]`.
- Produces `remote_cache_environment(config: ScratchConfig) -> dict[str, str]`.

- [ ] **Step 1: Write the failing dependency-light policy tests**

Create `tools/test_sitian_remote_scratch.py`:

```python
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "sitian_remote_scratch.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "sitian_remote_scratch_test_module",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PolicyTests(unittest.TestCase):
    def test_fixed_layout_stays_under_remote_task_root(self):
        module = load_module()
        config = module.ScratchConfig.default(ROOT)
        self.assertEqual(
            config.remote_root,
            "/data00/home/sitian/tinyllmforge-workspaces/"
            "command-timeline-20260818",
        )
        layout = module.remote_layout(config)
        self.assertEqual(
            set(layout),
            {"source", "tmp", "pycache", "cache", "logs", "receipts", "env"},
        )
        self.assertTrue(
            all(
                path.startswith(config.remote_root + "/")
                for path in layout.values()
            )
        )

    def test_explicit_paths_accept_only_clean_repository_relative_files(self):
        module = load_module()
        self.assertEqual(
            module.validate_relative_paths(
                [
                    "tools/sitian_remote_scratch.py",
                    "tools/test_sitian_remote_scratch.py",
                ]
            ),
            (
                "tools/sitian_remote_scratch.py",
                "tools/test_sitian_remote_scratch.py",
            ),
        )
        rejected = [
            "/private/tmp/output.log",
            "../TinyLLMForge-adaptive-ngram/file.py",
            ".git/config",
            "artifacts/run/output.json",
            "experiments/run/source.patch",
            "tools/__pycache__/module.pyc",
            ".superpowers/sdd/task-5-review-package.diff",
            "runner.log",
            "runner.pid",
        ]
        for path in rejected:
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    module.validate_relative_paths([path])

    def test_remote_cache_environment_has_no_local_tmp_path(self):
        module = load_module()
        config = module.ScratchConfig.default(ROOT)
        environment = module.remote_cache_environment(config)
        self.assertEqual(environment["TMPDIR"], config.remote_root + "/tmp")
        self.assertEqual(
            environment["PYTHONPYCACHEPREFIX"],
            config.remote_root + "/pycache",
        )
        self.assertEqual(
            environment["XDG_CACHE_HOME"],
            config.remote_root + "/cache",
        )
        self.assertNotIn("/tmp", "\n".join(environment.values()).replace(
            config.remote_root + "/tmp",
            "",
        ))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Stream the new test to a remote RED directory and verify failure**

Run from the authoritative checkout without creating a local archive:

```bash
cd /Users/bytedance/Desktop/TinyLLMForge
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -cf - \
  tools/test_sitian_remote_scratch.py |
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes \
  sitian@10.232.195.203 \
  'set -eu
   root=/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818
   red=$root/red-task1
   rm -rf "$red"
   mkdir -p "$red"
   tar -xf - -C "$red"
   cd "$red"
   PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 \
     tools/test_sitian_remote_scratch.py'
```

Expected: FAIL because `tools/sitian_remote_scratch.py` does not exist.

- [ ] **Step 3: Implement the fixed configuration and path policy**

Create `tools/sitian_remote_scratch.py` with:

```python
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence


REMOTE_HOST = "sitian@10.232.195.203"
REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
KRB5_CACHE = "FILE:/Users/bytedance/krb5cc_sitian"
FORBIDDEN_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "artifacts",
    "experiments",
}
FORBIDDEN_SUFFIXES = {
    ".pyc",
    ".log",
    ".pid",
    ".tar",
    ".tgz",
    ".gz",
    ".zip",
}


@dataclass(frozen=True)
class ScratchConfig:
    repo_root: Path
    remote_host: str = REMOTE_HOST
    remote_root: str = REMOTE_ROOT
    krb5_cache: str = KRB5_CACHE
    attempts: int = 5

    @classmethod
    def default(cls, repo_root: Path) -> "ScratchConfig":
        resolved = repo_root.resolve()
        expected = Path("/Users/bytedance/dev/TinyLLMForge")
        if resolved != expected:
            raise ValueError(
                "repo root must resolve to "
                "/Users/bytedance/dev/TinyLLMForge"
            )
        return cls(repo_root=repo_root)


def remote_layout(config: ScratchConfig) -> dict[str, str]:
    return {
        name: f"{config.remote_root}/{name}"
        for name in (
            "source",
            "tmp",
            "pycache",
            "cache",
            "logs",
            "receipts",
            "env",
        )
    }


def validate_relative_paths(paths: Sequence[str]) -> tuple[str, ...]:
    if not paths:
        raise ValueError("at least one explicit path is required")
    normalized = []
    for raw_path in paths:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("paths must be non-empty strings")
        path = PurePosixPath(raw_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"path is not repository-relative: {raw_path}")
        if any(part in FORBIDDEN_PARTS for part in path.parts):
            raise ValueError(f"path is forbidden: {raw_path}")
        if (
            path.suffix in FORBIDDEN_SUFFIXES
            or path.name.endswith("-review-package.diff")
        ):
            raise ValueError(f"path is forbidden: {raw_path}")
        normalized.append(path.as_posix())
    return tuple(dict.fromkeys(normalized))


def ssh_argv(config: ScratchConfig) -> tuple[str, ...]:
    return (
        "ssh",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
        config.remote_host,
    )


def remote_cache_environment(
    config: ScratchConfig,
) -> dict[str, str]:
    layout = remote_layout(config)
    return {
        "TMPDIR": layout["tmp"],
        "TMP": layout["tmp"],
        "TEMP": layout["tmp"],
        "PYTHONPYCACHEPREFIX": layout["pycache"],
        "XDG_CACHE_HOME": layout["cache"],
        "PYTHONDONTWRITEBYTECODE": "0",
    }
```

- [ ] **Step 4: Sync the implementation into the remote RED directory and verify GREEN**

Stream both files with the same xattr-disabled tar command and run:

```bash
cd /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/red-task1
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 \
  tools/test_sitian_remote_scratch.py
```

Expected: `Ran 3 tests` and `OK`.

- [ ] **Step 5: Commit and push Task 1**

```bash
git add -- \
  tools/sitian_remote_scratch.py \
  tools/test_sitian_remote_scratch.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(workflow): define sitian scratch policy" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 2: Streaming Initialization and Explicit Incremental Sync

**Files:**
- Modify: `tools/test_sitian_remote_scratch.py`
- Modify: `tools/sitian_remote_scratch.py`

**Interfaces:**
- Produces `run_with_retries(...) -> subprocess.CompletedProcess[str]`.
- Produces `initial_snapshot_commands(config) -> dict[str, tuple[str, ...] | str]`.
- Produces `incremental_sync_commands(config, paths)`.
- Produces CLI subcommands `init`, `sync`, and `status`.

- [ ] **Step 1: Add failing transport and retry tests**

Extend `tools/test_sitian_remote_scratch.py`:

```python
from unittest import mock


class TransportTests(unittest.TestCase):
    def test_retry_stops_after_first_success(self):
        module = load_module()
        runner = mock.Mock(side_effect=[
            module.subprocess.CompletedProcess(
                ["ssh"], 255, "", "Connection closed"
            ),
            module.subprocess.CompletedProcess(["ssh"], 0, "ok\n", ""),
        ])
        result = module.run_with_retries(
            ["ssh"],
            attempts=5,
            runner=runner,
            sleep=lambda _: None,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(runner.call_count, 2)

    def test_initial_snapshot_uses_git_archive_and_no_local_file(self):
        module = load_module()
        config = module.ScratchConfig.default(ROOT)
        commands = module.initial_snapshot_commands(config)
        self.assertEqual(
            commands["archive"],
            ("git", "archive", "--format=tar", "HEAD"),
        )
        self.assertIn(".incoming-", commands["remote_extract"])
        self.assertIn("find source -name '._*'", commands["remote_verify"])
        self.assertNotIn("/private/tmp", json.dumps(commands))

    def test_incremental_sync_requires_explicit_allowed_paths(self):
        module = load_module()
        config = module.ScratchConfig.default(ROOT)
        commands = module.incremental_sync_commands(
            config,
            ["tools/sitian_remote_scratch.py"],
        )
        self.assertIn("--no-xattrs", commands["tar"])
        self.assertIn("--no-mac-metadata", commands["tar"])
        self.assertIn(
            "tools/sitian_remote_scratch.py",
            commands["tar"],
        )
        with self.assertRaises(ValueError):
            module.incremental_sync_commands(
                config,
                [".superpowers/sdd/task-5-review-package.diff"],
            )
```

- [ ] **Step 2: Run the tests remotely and confirm RED**

Expected failures mention missing `run_with_retries`,
`initial_snapshot_commands`, and `incremental_sync_commands`.

- [ ] **Step 3: Implement bounded retry and transport commands**

Add:

```python
def run_with_retries(
    argv: Sequence[str],
    *,
    attempts: int,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs,
) -> subprocess.CompletedProcess[str]:
    last = None
    for attempt in range(1, attempts + 1):
        last = runner(argv, text=True, **kwargs)
        if last.returncode == 0:
            return last
        if attempt < attempts:
            sleep(2.0)
    assert last is not None
    return last


def initial_snapshot_commands(
    config: ScratchConfig,
) -> dict[str, tuple[str, ...] | str]:
    nonce = f"{os.getpid()}-{time.time_ns()}"
    stage = f"{config.remote_root}/.incoming-source-{nonce}"
    return {
        "archive": ("git", "archive", "--format=tar", "HEAD"),
        "stage": stage,
        "remote_extract": (
            "set -eu; "
            f"stage={shlex.quote(stage)}; "
            "rm -rf \"$stage\"; mkdir -p \"$stage/source\"; "
            "tar -xf - -C \"$stage/source\""
        ),
        "remote_verify": (
            "set -eu; "
            f"stage={shlex.quote(stage)}; "
            "cd \"$stage\"; "
            "test \"$(find source -name '._*' | wc -l)\" -eq 0; "
            "test \"$(find source -path "
            "'*/.superpowers/sdd/*review-package.diff' | wc -l)\" -eq 0"
        ),
    }


def incremental_sync_commands(
    config: ScratchConfig,
    paths: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    checked = validate_relative_paths(paths)
    return {
        "tar": (
            "tar",
            "--no-xattrs",
            "--no-mac-metadata",
            "-cf",
            "-",
            *checked,
        ),
        "ssh": (
            *ssh_argv(config),
            f"tar -xf - -C {shlex.quote(config.remote_root + '/source')}",
        ),
    }
```

Complete `init` so it:

1. resolves and records local `HEAD`;
2. creates a unique remote staging root;
3. streams `git archive --format=tar HEAD` to remote tar extraction;
4. verifies no `._*` or review-package files;
5. writes remote `receipts/source-head.txt` and a sorted
   `source-files.sha256`;
6. atomically promotes the staging source only when no verified `source/`
   exists, or promotes to `source-next/` and swaps only after verification;
7. leaves the last verified source unchanged on failure.

Complete `sync` so it:

1. accepts one or more explicit relative paths after `--`;
2. rejects missing, non-file, symlink-escaping, and forbidden paths;
3. streams an xattr-disabled tar directly to `source/`;
4. writes a timestamped remote path list and SHA-256 receipt; and
5. prints only the receipt path and transferred-file count locally.

- [ ] **Step 4: Run remote unit tests and verify GREEN**

Expected: all policy and transport tests pass with no files created under
local `/private/tmp`.

- [ ] **Step 5: Commit and push Task 2**

```bash
git add -- \
  tools/sitian_remote_scratch.py \
  tools/test_sitian_remote_scratch.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(workflow): stream source to sitian scratch" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 3: Remote Validation Runner, Logs, and Receipts

**Files:**
- Modify: `tools/test_sitian_remote_scratch.py`
- Modify: `tools/sitian_remote_scratch.py`

**Interfaces:**
- Produces `build_remote_run_script(config, run_id, command) -> str`.
- Produces CLI subcommand `run --id <run-id> -- <command...>`.
- Produces remote `<run-id>.log`, `<run-id>.exit`, and
  `<run-id>.receipt.json`.

- [ ] **Step 1: Add failing remote-run construction tests**

Add:

```python
class RemoteRunTests(unittest.TestCase):
    def test_remote_run_redirects_all_caches_and_full_output(self):
        module = load_module()
        config = module.ScratchConfig.default(ROOT)
        script = module.build_remote_run_script(
            config,
            "task5-focused",
            [
                "/usr/bin/python3",
                "tools/test_autoregressive_draft_command_timeline_diagnostic.py",
            ],
        )
        for value in module.remote_cache_environment(config).values():
            self.assertIn(value, script)
        self.assertIn(
            config.remote_root + "/logs/task5-focused.log",
            script,
        )
        self.assertIn(
            config.remote_root + "/receipts/task5-focused.receipt.json",
            script,
        )
        self.assertIn("tail -n 40", script)
        self.assertNotIn("/private/tmp", script)

    def test_run_id_rejects_shell_metacharacters(self):
        module = load_module()
        config = module.ScratchConfig.default(ROOT)
        with self.assertRaises(ValueError):
            module.build_remote_run_script(
                config,
                "task5;rm",
                ["/usr/bin/python3", "-V"],
            )
```

- [ ] **Step 2: Run the tests remotely and confirm RED**

Expected: failure because `build_remote_run_script` is missing.

- [ ] **Step 3: Implement the remote runner and bounded local output**

Add:

```python
def build_remote_run_script(
    config: ScratchConfig,
    run_id: str,
    command: Sequence[str],
) -> str:
    if (
        not run_id
        or any(
            character not in
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in run_id
        )
    ):
        raise ValueError("run id must contain only letters, digits, - and _")
    if not command:
        raise ValueError("remote command is required")
    layout = remote_layout(config)
    environment = remote_cache_environment(config)
    exports = " ".join(
        f"export {name}={shlex.quote(value)};"
        for name, value in environment.items()
    )
    log = f"{layout['logs']}/{run_id}.log"
    exit_path = f"{layout['receipts']}/{run_id}.exit"
    receipt = f"{layout['receipts']}/{run_id}.receipt.json"
    command_text = shlex.join(command)
    return (
        "set -u; "
        f"mkdir -p {shlex.quote(layout['tmp'])} "
        f"{shlex.quote(layout['pycache'])} "
        f"{shlex.quote(layout['cache'])} "
        f"{shlex.quote(layout['logs'])} "
        f"{shlex.quote(layout['receipts'])}; "
        f"{exports} "
        f"cd {shlex.quote(layout['source'])}; "
        f"( {command_text} ) >{shlex.quote(log)} 2>&1; "
        "rc=$?; "
        f"printf '%s\\n' \"$rc\" >{shlex.quote(exit_path)}; "
        f"sha=$(sha256sum {shlex.quote(log)} | awk '{{print $1}}'); "
        "python3 - \"$rc\" \"$sha\" "
        f"{shlex.quote(receipt)} <<'PY'\n"
        "import json, pathlib, sys\n"
        "rc, sha, target = sys.argv[1:]\n"
        "pathlib.Path(target).write_text(json.dumps({\n"
        "  'returncode': int(rc),\n"
        "  'log_sha256': sha,\n"
        "}, sort_keys=True) + '\\n')\n"
        "PY\n"
        f"tail -n 40 {shlex.quote(log)}; "
        "exit \"$rc\""
    )
```

The `run` CLI:

- checks that `source/` and the source receipt exist;
- executes through `run_with_retries` only until a remote command starts;
- never retries a command after receiving a remote exit status;
- prints at most 40 remote log lines plus the receipt path; and
- exits with the remote test return code.

- [ ] **Step 4: Run remote unit tests and verify GREEN**

Expected: all tests pass. Confirm the test itself creates no local matching
path under `/private/tmp`.

- [ ] **Step 5: Commit and push Task 3**

```bash
git add -- \
  tools/sitian_remote_scratch.py \
  tools/test_sitian_remote_scratch.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(workflow): run validations in sitian scratch" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 4: Real Remote Acceptance, Audit, and Task 5 Resume

**Files:**
- Create: `docs/superpowers/audits/2026-08-18-tinyllmforge-sitian-remote-scratch-audit.md`
- Modify: `.superpowers/sdd/progress.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes the `init`, `sync`, `status`, and `run` CLI commands.
- Produces a verified remote source snapshot and focused Task 5 test receipt.
- Produces the exact resume boundary for Task 5 independent re-review.

- [ ] **Step 1: Record the local no-cache baseline**

Run read-only checks:

```bash
find /private/tmp -maxdepth 1 -mindepth 1 \
  \( -iname '*tinyllmforge*' -o -iname '*command*timeline*' \) \
  -print
df -h /System/Volumes/Data
git status --short --branch
```

Expected: zero matching TinyLLMForge temporary items. Preserve unrelated dirty
worktree entries.

- [ ] **Step 2: Initialize the real remote source snapshot**

Run:

```bash
cd /Users/bytedance/Desktop/TinyLLMForge
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/sitian_remote_scratch.py init
```

Expected bounded local output:

```text
remote_source=/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/source
source_head=<current local HEAD>
appledouble_files=0
review_package_files=0
receipt=<remote receipt path>
```

- [ ] **Step 3: Synchronize the current focused Task 5 paths explicitly**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/sitian_remote_scratch.py sync -- \
  tools/autoregressive_draft_command_timeline_diagnostic.py \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py \
  .superpowers/sdd/progress.md
```

Expected: exactly three paths accepted. No review-package diff is transferred.

- [ ] **Step 4: Run focused Task 5 validation remotely**

Create the isolated remote test environment without mutating
`/data00/home/sitian/tllm/env`:

```bash
root=/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818
/data00/home/sitian/tllm/miniforge/bin/python -m venv "$root/env/test"
"$root/env/test/bin/python" -m pip install --disable-pip-version-check \
  pytest==8.4.2
"$root/env/test/bin/python" -c \
  'import pytest; assert pytest.__version__ == "8.4.2"'
```

Record the environment Python path and pytest version in a remote receipt.
The environment and pip cache remain under the remote task root.

The focused command is:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/sitian_remote_scratch.py run \
  --id task5-command-timeline-focused \
  -- /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/env/test/bin/python \
  -m pytest -p no:cacheprovider -q \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py
```

Expected: the previously verified Task 5 focused suite passes, the complete
log remains remote, and local output is limited to the final 40 lines.

- [ ] **Step 5: Verify remote containment and local non-creation**

Remote checks:

```bash
find /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818 \
  -name '._*' -print
find /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/source \
  -path '*/.superpowers/sdd/*review-package.diff' -print
cat /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/receipts/task5-command-timeline-focused.receipt.json
```

Local checks:

```bash
find /private/tmp -maxdepth 1 -mindepth 1 \
  \( -iname '*tinyllmforge*' -o -iname '*command*timeline*' \) \
  -print
git diff --check
```

Expected: both prohibited remote searches and the local temporary search are
empty; the receipt records return code zero and a log SHA-256.

- [ ] **Step 6: Write the acceptance audit and resume records**

The audit records:

- remote host and exact task root;
- local `HEAD` and source receipt;
- explicit synchronized paths;
- remote Python identity;
- focused command and return code;
- remote log and receipt SHA-256;
- zero `._*` and zero review-package files;
- before/after local `/private/tmp` matching count;
- no GPU or model execution; and
- Task 5 resume point: independent re-review of commits `b295e10` and
  `a1e6f42`, followed by Tasks 6-8 only if approved.

Append the same concise state to `.superpowers/sdd/progress.md` and
`AGENT_HANDOFF_STATE.md`.

- [ ] **Step 7: Verify, commit, and push the acceptance slice**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/sitian_remote_scratch.py status
git diff --check
git status --short -- \
  tools/sitian_remote_scratch.py \
  tools/test_sitian_remote_scratch.py \
  docs/superpowers/audits/2026-08-18-tinyllmforge-sitian-remote-scratch-audit.md \
  .superpowers/sdd/progress.md \
  AGENT_HANDOFF_STATE.md
```

Stage only:

```bash
git add -- \
  docs/superpowers/audits/2026-08-18-tinyllmforge-sitian-remote-scratch-audit.md \
  .superpowers/sdd/progress.md \
  AGENT_HANDOFF_STATE.md
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "docs(workflow): verify sitian scratch execution" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

- [ ] **Step 8: Resume the approved command-timeline optimization plan**

Return to
`docs/superpowers/plans/2026-08-18-autoregressive-draft-command-timeline-sync-debt.md`.
Use the sitian scratch runner for every cache-producing test or review command.
Complete Task 5 independent re-review before starting Task 6. Do not run the
remote GPU gate without separate explicit authorization.
