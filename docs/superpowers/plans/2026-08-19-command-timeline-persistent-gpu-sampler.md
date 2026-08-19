# Command-Timeline Persistent GPU Sampler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-sample `nvidia-smi` process creation with one selected-GPU persistent sampler while preserving strict in-repeat four-GPU telemetry admission.

**Architecture:** Keep orchestration and artifact schemas unchanged. Generate a dependency-light Python sampler program from the remote runner; that program owns one `nvidia-smi --loop-ms=200` child, groups exactly one row per selected GPU into a canonical snapshot, timestamps the completed snapshot, and reaps the child on every exit path.

**Tech Stack:** Python 3.11, `subprocess.Popen`, POSIX signals, `nvidia-smi`, JSON Lines, pytest 8.4.2, existing Sitian source transaction engine.

## Global Constraints

- Modify only `/Users/bytedance/Desktop/TinyLLMForge`.
- Keep branch `feat/kv-sparse-attention`.
- Never modify `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Preserve immutable failed campaign tags `20260818-command-timeline-tp4-b4-q4-r6` and `20260818-command-timeline-tp4-b4-q4-r7`.
- Use only the fresh tag `20260818-command-timeline-tp4-b4-q4-r8` for the next campaign.
- Keep every generated remote environment, cache, basetemp, log, receipt, manifest, review, and campaign artifact beneath `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Do not create task artifacts under local or remote `/`, `/tmp`, `/private/tmp`, or repository source.
- Do not refresh Kerberos manually.
- Do not terminate, pause, signal, adopt, or interfere with unrelated GPU or host processes.
- Keep the four-clean-GPU gate unchanged: four distinct GPUs, memory used at most 1024 MiB, utilization at most 5%, and no compute process.
- Keep strict telemetry admission unchanged: every measured repeat requires an in-interval complete four-GPU snapshot and an in-interval host row.
- Do not borrow boundary samples, duplicate samples, weaken verifiers, change runtime semantics, or authorize a runtime optimization.
- Stage only explicit paths; never use `git add -A`.
- Commit with `git -c core.hooksPath=/dev/null commit` and exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Push each completed versionable slice to `origin/feat/kv-sparse-attention`.

---

### Task 1: Persistent Sampler RED Contracts

**Files:**
- Modify: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Consumes: `_load_command_timeline_runner(module_name)`.
- Produces tests for: `_build_gpu_sampler_script() -> str` and `_start_epoch_samplers(...)`.

- [x] **Step 1: Add a fake `nvidia-smi` helper**

Add:

```python
def _write_fake_nvidia_smi(tmp_path, body):
    path = tmp_path / "nvidia-smi"
    path.write_text(
        "#!/usr/bin/env python3\n" + body,
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _run_gpu_sampler_script(runner, tmp_path, body):
    _write_fake_nvidia_smi(tmp_path, body)
    environment = os.environ.copy()
    environment["PATH"] = (
        str(tmp_path) + os.pathsep + environment.get("PATH", "")
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            runner._build_gpu_sampler_script(),
            json.dumps([2, 3, 4, 6]),
        ],
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
```

- [x] **Step 2: Add complete-snapshot and command-identity tests**

Add:

```python
def test_command_timeline_persistent_gpu_sampler_emits_complete_snapshots(
    tmp_path,
):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_persistent_gpu_sampler_test"
    )
    arguments_path = tmp_path / "arguments.json"
    body = f"""
import json
import os
from pathlib import Path
import sys

Path({str(arguments_path)!r}).write_text(
    json.dumps(sys.argv[1:]),
    encoding="utf-8",
)
for snapshot in range(2):
    for index in (2, 3, 4, 6):
        print(
            "2026/08/19 18:00:00.000, "
            f"{{index}}, GPU-{{index}}, P0, 1410, 1512, "
            "70.0, 40, 50, 10, 100, 0x0",
            flush=True,
        )
"""
    result = _run_gpu_sampler_script(runner, tmp_path, body)

    assert result.returncode == 0, result.stderr
    rows = [
        json.loads(line)
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    assert len(rows) == 8
    for offset in (0, 4):
        snapshot = rows[offset:offset + 4]
        assert [row["gpu_index"] for row in snapshot] == [2, 3, 4, 6]
        assert len({
            row["sampled_at_unix_ns"] for row in snapshot
        }) == 1
        assert len({
            row["sampled_at_monotonic_ns"] for row in snapshot
        }) == 1

    arguments = json.loads(arguments_path.read_text(encoding="utf-8"))
    assert "--id=2,3,4,6" in arguments
    assert "--loop-ms=200" in arguments
    assert "--format=csv,noheader,nounits" in arguments
    assert any(
        argument.startswith("--query-gpu=timestamp,index,uuid,")
        for argument in arguments
    )
```

- [x] **Step 3: Add fail-closed partial and duplicate tests**

Add:

```python
@pytest.mark.parametrize(
    ("indices", "message"),
    (
        ((2, 3, 4), "incomplete GPU snapshot"),
        ((2, 3, 2), "duplicate GPU row"),
    ),
)
def test_command_timeline_persistent_gpu_sampler_rejects_bad_snapshots(
    tmp_path,
    indices,
    message,
):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_bad_gpu_snapshot_test"
        + message.replace(" ", "_")
    )
    body = """
for index in %r:
    print(
        "2026/08/19 18:00:00.000, "
        f"{index}, GPU-{index}, P0, 1410, 1512, "
        "70.0, 40, 50, 10, 100, 0x0",
        flush=True,
    )
""" % (indices,)
    result = _run_gpu_sampler_script(runner, tmp_path, body)

    assert result.returncode != 0
    assert message in result.stderr
    assert result.stdout == ""
```

- [x] **Step 4: Add owned-child reaping test**

Add:

```python
def test_command_timeline_persistent_gpu_sampler_reaps_child_on_sigterm(
    tmp_path,
):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_gpu_sampler_reap_test"
    )
    pid_path = tmp_path / "nvidia-smi.pid"
    _write_fake_nvidia_smi(
        tmp_path,
        f"""
from pathlib import Path
import os
import time

Path({str(pid_path)!r}).write_text(str(os.getpid()), encoding="utf-8")
while True:
    time.sleep(1)
""",
    )
    environment = os.environ.copy()
    environment["PATH"] = (
        str(tmp_path) + os.pathsep + environment.get("PATH", "")
    )
    sampler = subprocess.Popen(
        [
            sys.executable,
            "-c",
            runner._build_gpu_sampler_script(),
            json.dumps([2, 3, 4, 6]),
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while not pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pid_path.exists()
    child_pid = int(pid_path.read_text(encoding="utf-8"))

    sampler.terminate()
    sampler.wait(timeout=5)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail("owned nvidia-smi child was not reaped")
```

- [x] **Step 5: Add strict coverage regression**

Add:

```python
def test_command_timeline_telemetry_attachment_rejects_boundary_only_gpu_rows(
    tmp_path,
):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_strict_telemetry_coverage_test"
    )
    start = 1_800_000_000_000_000_000
    finish = start + 1_000_000_000
    gpu_uuids = [f"GPU-{index}" for index in range(4)]
    gpu_path = tmp_path / "gpu.jsonl"
    host_path = tmp_path / "host.jsonl"
    gpu_path.write_text(
        "".join(
            json.dumps({
                "sampled_at_unix_ns": start - 1,
                "sampled_at_monotonic_ns": 1,
                "gpu_uuid": uuid,
            }) + "\n"
            for uuid in gpu_uuids
        ),
        encoding="utf-8",
    )
    host_path.write_text(
        json.dumps({
            "sampled_at_unix_ns": start + 1,
            "sampled_at_monotonic_ns": 2,
        }) + "\n",
        encoding="utf-8",
    )
    worker = {
        "measured_runs": [
            {
                "campaign_interval": {
                    "started_at_unix_ns": start + repeat * 2_000_000_000,
                    "finished_at_unix_ns": (
                        finish + repeat * 2_000_000_000
                    ),
                },
                "command_timeline_repeat_index": repeat,
            }
            for repeat in range(5)
        ],
    }

    with pytest.raises(
        ValueError,
        match="telemetry coverage is incomplete",
    ):
        runner._attach_epoch_telemetry(
            worker,
            gpu_path=gpu_path,
            host_path=host_path,
            gpu_uuids=gpu_uuids,
        )
```

- [x] **Step 6: Run focused tests and verify RED**

Atomically sync only the modified test file to the remote authoritative
source, then run:

```bash
cd /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/source
export TMPDIR=/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/tmp
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export PYTHONPYCACHEPREFIX=/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/pycache
export XDG_CACHE_HOME=/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/cache
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/env/test/bin/python \
  -m pytest -q \
  -p no:cacheprovider \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k 'persistent_gpu_sampler or boundary_only_gpu_rows' \
  --basetemp /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/pytest/persistent-gpu-sampler-red
```

Expected: the persistent sampler tests fail because
`_build_gpu_sampler_script` does not exist. The strict boundary-only test
passes, proving admission has not already been weakened.

---

### Task 2: Persistent Sampler GREEN Implementation

**Files:**
- Modify: `tools/run_autoregressive_draft_command_timeline_remote.py`
- Test: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Produces: `_build_gpu_sampler_script() -> str`.
- Preserves: `_start_epoch_samplers(...) -> tuple[list[subprocess.Popen], list[object]]`.

- [x] **Step 1: Add `_build_gpu_sampler_script`**

Place it immediately before `_start_epoch_samplers`:

```python
def _build_gpu_sampler_script() -> str:
    return "\n".join((
        "import csv,json,signal,subprocess,sys,time",
        "indices=json.loads(sys.argv[1])",
        "if (not isinstance(indices,list) or len(indices)!=4",
        " or any(isinstance(index,bool) or not isinstance(index,int)",
        " or index<0 for index in indices)",
        " or len(set(indices))!=4):",
        " raise ValueError('GPU indices are invalid')",
        "expected=tuple(indices)",
        "stop_requested=False",
        "child=None",
        "def request_stop(_signum,_frame):",
        " global stop_requested",
        " stop_requested=True",
        " if child is not None and child.poll() is None:",
        "  child.terminate()",
        "signal.signal(signal.SIGTERM,request_stop)",
        "signal.signal(signal.SIGINT,request_stop)",
        "command=[",
        " 'nvidia-smi',",
        " '--id='+','.join(str(index) for index in expected),",
        " '--query-gpu=timestamp,index,uuid,pstate,"
        "clocks.current.sm,clocks.current.memory,power.draw,"
        "temperature.gpu,utilization.gpu,utilization.memory,"
        "memory.used,clocks_throttle_reasons.active',",
        " '--format=csv,noheader,nounits',",
        " '--loop-ms=200']",
        "try:",
        " child=subprocess.Popen(command,stdout=subprocess.PIPE,",
        "  text=True,bufsize=1)",
        " if child.stdout is None:",
        "  raise RuntimeError('nvidia-smi stdout is unavailable')",
        " snapshot={}",
        " for fields in csv.reader(child.stdout,skipinitialspace=True):",
        "  if stop_requested:",
        "   break",
        "  fields=[part.strip() for part in fields]",
        "  if not fields:",
        "   continue",
        "  if len(fields)!=12:",
        "   raise ValueError('GPU telemetry field count is invalid')",
        "  index=int(fields[1])",
        "  if index not in expected:",
        "   raise ValueError('unexpected GPU row')",
        "  if index in snapshot:",
        "   raise ValueError('duplicate GPU row')",
        "  snapshot[index]=fields",
        "  if len(snapshot)!=len(expected):",
        "   continue",
        "  unix_ns=time.time_ns()",
        "  monotonic_ns=time.monotonic_ns()",
        "  for selected in expected:",
        "   row=snapshot[selected]",
        "   print(json.dumps({",
        "    'sampled_at_unix_ns':unix_ns,",
        "    'sampled_at_monotonic_ns':monotonic_ns,",
        "    'nvidia_timestamp':row[0],",
        "    'gpu_index':int(row[1]),",
        "    'gpu_uuid':row[2],",
        "    'pstate':row[3],",
        "    'sm_clock_mhz':int(row[4]),",
        "    'memory_clock_mhz':int(row[5]),",
        "    'power_w':float(row[6]),",
        "    'temperature_c':int(row[7]),",
        "    'gpu_utilization_percent':int(row[8]),",
        "    'memory_utilization_percent':int(row[9]),",
        "    'memory_used_mib':int(row[10]),",
        "    'throttle_reasons_active':int(row[11],0)},",
        "    sort_keys=True,separators=(',',':')),flush=True)",
        "  snapshot={}",
        " returncode=child.wait()",
        " if stop_requested:",
        "  raise SystemExit(0)",
        " if snapshot:",
        "  raise ValueError('incomplete GPU snapshot')",
        " if returncode:",
        "  raise SystemExit(returncode)",
        "finally:",
        " if child is not None:",
        "  if child.poll() is None:",
        "   child.terminate()",
        "  try:",
        "   child.wait(timeout=5)",
        "  except subprocess.TimeoutExpired:",
        "   child.kill()",
        "   child.wait(timeout=5)",
    ))
```

- [x] **Step 2: Use the generated persistent program**

Replace the inline repeated-subprocess `gpu_script` in
`_start_epoch_samplers(...)` with:

```python
    gpu_script = _build_gpu_sampler_script()
```

Keep the existing outer sampler launch, file handles, stderr capture, host
sampler command, and cleanup unchanged.

- [x] **Step 3: Run focused tests and verify GREEN**

Atomically sync the runner and test file to the remote authoritative source,
then run:

```bash
cd /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/source
export TMPDIR=/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/tmp
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export PYTHONPYCACHEPREFIX=/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/pycache
export XDG_CACHE_HOME=/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/cache
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/env/test/bin/python \
  -m pytest -q \
  -p no:cacheprovider \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k 'persistent_gpu_sampler or boundary_only_gpu_rows or epoch_samplers' \
  --basetemp /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/pytest/persistent-gpu-sampler-green
```

Expected: all selected tests pass.

- [x] **Step 4: Run the full remote runner/diagnostic contract**

Run:

```bash
cd /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/source
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/env/test/bin/python \
  -m pytest -q \
  -p no:cacheprovider \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  tools/test_autoregressive_draft_performance_gate.py \
  --basetemp /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/pytest/persistent-gpu-sampler-contract
```

Expected: all tests pass, including ownership-race and strict telemetry
coverage tests.

- [x] **Step 5: Run syntax and diff checks**

Run syntax validation remotely with task-root cache variables, then run the
local diff check:

```bash
cd /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/source
PYTHONPYCACHEPREFIX=/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/pycache \
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/env/test/bin/python \
  -m py_compile \
  tools/run_autoregressive_draft_command_timeline_remote.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py

cd /Users/bytedance/Desktop/TinyLLMForge
git diff --check -- \
  tools/run_autoregressive_draft_command_timeline_remote.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py
```

Expected: both commands exit zero.

- [ ] **Step 6: Commit and push the implementation**

Run:

```bash
git add \
  tools/run_autoregressive_draft_command_timeline_remote.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py
git -c core.hooksPath=/dev/null commit \
  -m "fix(command-timeline): persist GPU telemetry sampler" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

Expected: local and origin branch heads match the new commit.

---

### Task 3: Remote Acceptance and Fresh r8 Campaign

**Files:**
- Modify after verified campaign:
  `docs/superpowers/audits/2026-08-18-autoregressive-draft-command-timeline-local-audit.md`
- Modify after verified campaign: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: pushed implementation commit and the existing atomic Sitian
  source transaction engine.
- Produces: immutable r8 primary/controller artifacts, dual receipts,
  checksum manifest, reconciled audit, and handoff.

- [ ] **Step 1: Atomically sync the two changed source files**

Use the task-owned SSH channel and the transaction engine to sync exactly:

```text
tools/run_autoregressive_draft_command_timeline_remote.py
tools/test_autoregressive_draft_cuda_graph_gate.py
```

Verify:

- remote source head equals local/origin head;
- detached sync receipt exists;
- receipt SHA-256 values equal local file SHA-256 values.

- [ ] **Step 2: Run remote focused and full contract tests**

Use the pinned pytest environment under the Sitian task root:

```bash
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/env/test/bin/python \
  -m pytest -q \
  -p no:cacheprovider \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k 'persistent_gpu_sampler or boundary_only_gpu_rows or epoch_samplers'
```

Then run the same three-file full contract from Task 2 Step 4. Set
`TMPDIR`, `TMP`, `TEMP`, `PYTHONPYCACHEPREFIX`, `XDG_CACHE_HOME`, and
`--basetemp` beneath the Sitian task root.

- [ ] **Step 3: Verify fresh tag destinations**

Confirm both paths are absent:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/runs/20260818-command-timeline-tp4-b4-q4-r8
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/controller-verification/20260818-command-timeline-tp4-b4-q4-r8
```

- [ ] **Step 4: Start the Mac-Agent automatic controller**

The controller must:

1. poll official `preflight --run-tag 20260818-command-timeline-tp4-b4-q4-r8`
   every 15 seconds;
2. retry transient transport errors;
3. stop fail-fast on insufficient Kerberos lifetime;
4. call official `execute` immediately on `READY`;
5. never pause for another user response;
6. retain execute's built-in second preflight.

- [ ] **Step 5: Verify the completed campaign**

Require all eight epoch identities:

```text
block-0:eager:first
block-0:graph:second
block-1:graph:first
block-1:eager:second
block-2:graph:first
block-2:eager:second
block-3:eager:first
block-3:graph:second
```

Require:

- `status=PASS`;
- primary remote verifier PASS;
- controller verifier PASS;
- normalized receipts equal;
- `manifest.sha256` validates;
- canonical result classification is read from verified artifacts;
- runtime optimization remains unauthorized unless the verified result is
  `BOUNDARY_LOCALIZED` and a separate design is approved.

- [ ] **Step 6: Reconcile canonical documentation**

Append one chronological reconciliation to the audit and handoff that
records:

- `r6`: GPU ownership snapshot race, immutable failure;
- commit `d975a30`: ownership fix and regression results;
- `r7`: successful worker/ownership, failed strict telemetry coverage;
- persistent sampler design, tests, commit, and remote sync receipt;
- `r8`: exact GPUs, eight epochs, verifier/manifest evidence, and final
  classification;
- explicit claim boundaries and next authorized action.

- [ ] **Step 7: Final verification, commit, and push**

Run focused documentation checks, `git diff --check`, exact-path status,
and inspect the final diff. Stage only the audit and handoff:

```bash
git add \
  docs/superpowers/audits/2026-08-18-autoregressive-draft-command-timeline-local-audit.md \
  AGENT_HANDOFF_STATE.md
git -c core.hooksPath=/dev/null commit \
  -m "docs(command-timeline): reconcile persistent sampler campaign" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

Expected: local, remote source, and origin commits are reconciled; no
unrelated path is staged or modified.
