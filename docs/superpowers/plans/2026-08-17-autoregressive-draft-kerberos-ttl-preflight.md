# Autoregressive Draft Kerberos TTL Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject a long TP4/B4/Q4 remote CUDA Graph campaign before any local-run or SSH side effect when the expected Kerberos TGT has less than 5,400 seconds remaining.

**Architecture:** Add a pure Kerberos JSON classifier and a small subprocess adapter to the existing remote runner. `execute_remote_gate()` checks local authentication before creating `local_run`, while preflight-only mode checks it before any SSH command; READY attempts then preserve the current source-bound remote workflow and record only normalized, non-secret Kerberos metadata.

**Tech Stack:** Python 3.11 standard library (`datetime`, `json`, `subprocess`), pytest 8.4.2, TinyLLMForge source-bound remote gate.

## Global Constraints

- The expected client principal is exactly `sitian@BYTEDANCE.COM`.
- The expected TGT principal is exactly `krbtgt/BYTEDANCE.COM@BYTEDANCE.COM`.
- The minimum remaining TGT lifetime is exactly `5400` seconds.
- Parse Kerberos timestamps in `YYYYMMDDHHMMSS` form using the timezone of the injected current time.
- Local authentication failure is `INCONCLUSIVE_ENVIRONMENT`, never a CUDA Graph correctness or performance failure.
- Reject local authentication before creating a local run directory, reserving a run tag, or issuing an SSH command.
- Never run `kinit`, prompt for or store a password, read Keychain data, detach the gate, resume an interrupted tag, or terminate any local or remote process.
- Do not write credential bytes, the raw `klist` payload, or cache contents into artifacts.
- Preserve exact TP4/B4/Q4, prompt length 256, output length 16, exact-greedy tokens, Proposal-KV transaction semantics, and TP failure convergence.
- Use explicit file staging; never use `git add -A`.
- Every commit ends with exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.

---

### Task 1: Classify Local Kerberos Lifetime

**Files:**
- Modify: `tools/run_autoregressive_draft_cuda_graph_gate_remote.py`
- Test: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Consumes: `klist --json` decoded mapping, timezone-aware `datetime`, expected principal/TGT, and minimum lifetime.
- Produces: `classify_local_kerberos_payload(payload, *, now, minimum_lifetime_seconds=5400) -> dict`.
- Produces: `_local_kerberos_preflight(*, command_runner=subprocess.run, now=None) -> dict`.

- [ ] **Step 1: Add failing pure-classifier tests**

Add imports:

```python
from datetime import datetime
from zoneinfo import ZoneInfo
```

Add a fixture helper:

```python
def _kerberos_payload(*, expires, principal="sitian@BYTEDANCE.COM"):
    return {
        "version": 1,
        "cache": "API:redacted",
        "principal": principal,
        "tickets": [{
            "Issued": "20260817120000",
            "Expires": expires,
            "Principal": "krbtgt/BYTEDANCE.COM@BYTEDANCE.COM",
        }],
    }
```

Use `now = datetime(2026, 8, 17, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))`
and assert:

```python
ready = runner.classify_local_kerberos_payload(
    _kerberos_payload(expires="20260817220001"),
    now=now,
)
assert ready == {
    "status": "READY",
    "principal": "sitian@BYTEDANCE.COM",
    "tgt_principal": "krbtgt/BYTEDANCE.COM@BYTEDANCE.COM",
    "expires_at": "2026-08-17T22:00:01+08:00",
    "remaining_lifetime_seconds": 7201,
    "minimum_required_lifetime_seconds": 5400,
}
```

Add parametrized cases for:

```text
Expires=20260817195959 -> local Kerberos TGT is expired
Expires=20260817212959 -> local Kerberos TGT lifetime is insufficient
no matching TGT       -> local Kerberos TGT is missing
wrong client          -> local Kerberos principal is unexpected
non-mapping payload   -> local Kerberos payload is invalid
malformed Expires     -> local Kerberos payload is invalid
```

Each rejected result must have:

```python
assert result["status"] == "INCONCLUSIVE_ENVIRONMENT"
assert result["reason"] == expected_reason
assert result["minimum_required_lifetime_seconds"] == 5400
assert "cache" not in result
```

- [ ] **Step 2: Run classifier tests and verify RED**

Run:

```bash
uv run --offline --python 3.11 --with pytest==8.4.2 \
  pytest -q tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k local_kerberos_payload
```

Expected: failure because `classify_local_kerberos_payload` does not exist.

- [ ] **Step 3: Implement the pure classifier**

Add constants:

```python
from datetime import datetime

EXPECTED_KERBEROS_PRINCIPAL = "sitian@BYTEDANCE.COM"
EXPECTED_KERBEROS_TGT = "krbtgt/BYTEDANCE.COM@BYTEDANCE.COM"
MINIMUM_KERBEROS_LIFETIME_SECONDS = 5400
KERBEROS_TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
```

Implement:

```python
def _kerberos_inconclusive(reason, *, minimum, principal=None,
                           tgt_principal=None, expires_at=None,
                           remaining=None):
    result = {
        "status": "INCONCLUSIVE_ENVIRONMENT",
        "reason": reason,
        "minimum_required_lifetime_seconds": minimum,
    }
    if principal is not None:
        result["principal"] = principal
    if tgt_principal is not None:
        result["tgt_principal"] = tgt_principal
    if expires_at is not None:
        result["expires_at"] = expires_at
    if remaining is not None:
        result["remaining_lifetime_seconds"] = remaining
    return result
```

`classify_local_kerberos_payload()` must:

1. reject non-mappings, non-list `tickets`, naive `now`, malformed ticket rows,
   and malformed timestamps as `local Kerberos payload is invalid`;
2. reject a client mismatch as `local Kerberos principal is unexpected`;
3. select the ticket whose `Principal` exactly equals
   `EXPECTED_KERBEROS_TGT`;
4. attach `now.tzinfo` to the parsed compact expiration;
5. compute `int((expires_at - now).total_seconds())`;
6. classify `remaining <= 0` as expired and `0 < remaining < minimum` as
   insufficient; and
7. return only normalized identity, ISO timestamp, lifetime, and threshold.

- [ ] **Step 4: Add failing subprocess-adapter tests**

Inject a fake command runner and assert the exact command is:

```python
["klist", "--json"]
```

For a nonzero return code, assert:

```python
assert result == {
    "status": "INCONCLUSIVE_ENVIRONMENT",
    "reason": "local Kerberos cache is unavailable",
    "minimum_required_lifetime_seconds": 5400,
}
```

For invalid stdout JSON, assert reason
`local Kerberos payload is invalid`. For valid stdout, assert the adapter
returns the pure classifier result.

- [ ] **Step 5: Run adapter tests and verify RED**

Run:

```bash
uv run --offline --python 3.11 --with pytest==8.4.2 \
  pytest -q tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k local_kerberos_preflight
```

Expected: failure because `_local_kerberos_preflight` does not exist.

- [ ] **Step 6: Implement the subprocess adapter**

Implement:

```python
def _local_kerberos_preflight(
    *,
    command_runner=subprocess.run,
    now=None,
) -> dict:
    current = datetime.now().astimezone() if now is None else now
    result = command_runner(
        ["klist", "--json"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return _kerberos_inconclusive(
            "local Kerberos cache is unavailable",
            minimum=MINIMUM_KERBEROS_LIFETIME_SECONDS,
        )
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError):
        return _kerberos_inconclusive(
            "local Kerberos payload is invalid",
            minimum=MINIMUM_KERBEROS_LIFETIME_SECONDS,
        )
    return classify_local_kerberos_payload(payload, now=current)
```

Do not include stderr/stdout or the raw cache name in the returned mapping.

- [ ] **Step 7: Run Task 1 tests and verify GREEN**

Run:

```bash
uv run --offline --python 3.11 --with pytest==8.4.2 \
  pytest -q tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k 'local_kerberos_payload or local_kerberos_preflight'
```

Expected: all selected tests pass.

---

### Task 2: Enforce the Guard Before Local and Remote Side Effects

**Files:**
- Modify: `tools/run_autoregressive_draft_cuda_graph_gate_remote.py`
- Test: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Consumes: `_local_kerberos_preflight()` from Task 1 and the existing
  `_remote_preflight()`.
- Produces: `_preflight_only(..., kerberos_command_runner, command_runner, now) -> dict`.
- Extends: `execute_remote_gate(..., kerberos_command_runner=subprocess.run, now=None) -> dict`.

- [ ] **Step 1: Add a failing preflight-only no-SSH test**

Use separate fake runners:

```python
kerberos_commands = []
remote_commands = []

def expired_kerberos(command, **_kwargs):
    kerberos_commands.append(command)
    return types.SimpleNamespace(
        returncode=0,
        stdout=json.dumps(_kerberos_payload(expires="20260817195959")),
        stderr="",
    )

def forbidden_remote(command, **_kwargs):
    remote_commands.append(command)
    raise AssertionError("SSH must not run")
```

Call `_preflight_only()` with the fixed `now` and assert:

```python
assert result["status"] == "INCONCLUSIVE_ENVIRONMENT"
assert result["reason"] == "local Kerberos TGT is expired"
assert kerberos_commands == [["klist", "--json"]]
assert remote_commands == []
```

- [ ] **Step 2: Add a failing execute no-side-effect test**

Call `execute_remote_gate()` with an expired Kerberos runner, a remote runner
that raises if called, and `local_run=tmp_path / "must-not-exist"`. Assert:

```python
assert result["classification"] == "INCONCLUSIVE_ENVIRONMENT"
assert result["preflight"]["reason"] == "local Kerberos TGT is expired"
assert not (tmp_path / "must-not-exist").exists()
assert remote_commands == []
```

- [ ] **Step 3: Add a failing READY ordering test**

Use a valid Kerberos payload and a remote runner that records whether
`local_run` exists at its first SSH invocation. Return a remote
`INCONCLUSIVE_ENVIRONMENT` GPU preflight so the test ends before source
staging. Assert:

```python
assert observations == ["local-run-exists-before-ssh"]
preflight = json.loads((local_run / "preflight.json").read_text())
assert preflight["local_kerberos"]["status"] == "READY"
assert "cache" not in preflight["local_kerberos"]
```

Also assert the existing remote exact TP4/B4/Q4 arguments are unchanged in
`test_remote_command_binds_exact_tp4_b4_q4_gate`.

- [ ] **Step 4: Run ordering tests and verify RED**

Run:

```bash
uv run --offline --python 3.11 --with pytest==8.4.2 \
  pytest -q tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k 'preflight_only_avoids_ssh or execute_remote_gate_avoids_side_effects or ready_kerberos_precedes_remote'
```

Expected: failures because the runner currently creates `local_run` and
contacts SSH without the local guard.

- [ ] **Step 5: Implement preflight-only composition**

Implement `_preflight_only()` so it:

1. runs `_local_kerberos_preflight()` first;
2. returns that mapping immediately when status is not READY;
3. calls `_remote_preflight()` only when local status is READY; and
4. returns remote fields plus `local_kerberos`.

Update `main()` preflight-only mode to call `_preflight_only()`.

- [ ] **Step 6: Reorder execute_remote_gate**

Before `local_run.mkdir(...)`, run:

```python
local_kerberos = _local_kerberos_preflight(
    command_runner=kerberos_command_runner,
    now=now,
)
if local_kerberos["status"] != "READY":
    return {
        "classification": "INCONCLUSIVE_ENVIRONMENT",
        "preflight": local_kerberos,
        "local_run": str(local_run),
    }
```

Only then create `local_run`, run `_remote_preflight()`, combine:

```python
preflight = {
    **remote_preflight,
    "local_kerberos": local_kerberos,
}
```

and write normalized `preflight.json`. Keep all source staging, foreground
gate, remote verifier, checksum manifest, download, and local verifier logic
unchanged.

- [ ] **Step 7: Run Task 2 tests and verify GREEN**

Run the Step 4 command. Expected: all selected tests pass.

- [ ] **Step 8: Run the complete focused regression suite**

Run:

```bash
uv run --offline --python 3.11 \
  --with pytest==8.4.2 \
  --with torch==2.7.1 \
  --with transformers==4.57.6 \
  --with numpy \
  pytest -q \
    tools/test_autoregressive_draft_cuda_graph_gate.py \
    tools/test_autoregressive_draft_performance_gate.py \
    tools/test_autoregressive_draft_graph.py \
    tools/test_qwen3_draft_cuda_graph_backend.py
```

Expected: all tests pass with no deselected failure hidden.

- [ ] **Step 9: Run syntax, source-safety, and whitespace checks**

Run:

```bash
uv run --offline --python 3.11 python -m compileall -q \
  tools/run_autoregressive_draft_cuda_graph_gate_remote.py \
  tools/autoregressive_draft_cuda_graph_gate.py \
  tools/autoregressive_draft_cuda_graph_contract.py \
  tools/verify_autoregressive_draft_cuda_graph_gate.py
pytest -q tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k remote_runner_source_has_no_process_destruction
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 10: Commit and push the guard**

Stage only:

```bash
git add \
  docs/superpowers/plans/2026-08-17-autoregressive-draft-kerberos-ttl-preflight.md \
  tools/run_autoregressive_draft_cuda_graph_gate_remote.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py
```

Commit and push:

```bash
git -c core.hooksPath=/dev/null commit \
  -m "fix(cuda-graph): fail fast on short Kerberos TTL" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

Expected: local and remote branch heads match.

---

### Task 3: Run the Fresh Source-Bound Schema-v2 Gate

**Files:**
- Create but never stage: `artifacts/autoregressive_draft_cuda_graph/<new-run-tag>/`
- Modify after complete evidence exists: `docs/superpowers/audits/2026-08-17-autoregressive-draft-cuda-graph-completion-audit.md`
- Modify after complete evidence exists: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: a user-renewed Kerberos credential with at least 5,400 seconds
  remaining, an owned SSH ControlMaster, four clean GPUs, and the committed
  TTL guard.
- Produces: two warmup pairs, eight measured balanced pairs, remote and local
  verifier receipts, checksum manifest, final classification, audit, and
  handoff.

- [ ] **Step 1: Rebuild and verify the owned ControlMaster**

After the user renews Kerberos externally, run:

```bash
ssh -MNf \
  -o ControlMaster=yes \
  -o ControlPath=/tmp/ssh-sitian-10.232.195.203 \
  -o ControlPersist=4h \
  sitian@10.232.195.203
ssh -S /tmp/ssh-sitian-10.232.195.203 \
  -O check sitian@10.232.195.203
```

Expected: the control socket reports a running master owned by this task.

- [ ] **Step 2: Inspect interrupted state without mutation**

Read the interrupted remote run directory, active process list, and
`nvidia-smi` state. Do not kill, pause, delete, resume, or overwrite anything.
Classify the old tag only as `INCONCLUSIVE_ENVIRONMENT_AUTH_EXPIRED`.

- [ ] **Step 3: Run preflight-only with the TTL guard**

Run:

```bash
uv run --offline --python 3.11 \
  tools/run_autoregressive_draft_cuda_graph_gate_remote.py \
  --preflight-only
```

Expected: local Kerberos status READY, remote prerequisites READY, and four
clean selected GPUs. If not READY, stop before creating a new artifact tag.

- [ ] **Step 4: Start a completely new foreground tag**

Choose a new tag that has never existed locally or remotely. Run
`tools/run_autoregressive_draft_cuda_graph_gate_remote.py` in the foreground
with the new tag and a matching path under
`artifacts/autoregressive_draft_cuda_graph/`.

- [ ] **Step 5: Verify and classify complete evidence**

Require all of:

```text
two pair-level warmup pairs
eight measured pairs with four eager_graph and four graph_eager
exact target/proposal/accepted-prefix/transaction parity
zero active transactions, fallback, and quarantine
capture/resource stability and replay growth on every rank
acceptance and phase timing evidence
valid source_manifest.json and manifest.sha256
verify.remote.json and verify.local.json
```

Only complete evidence may produce `GO`, `NO_GO_CORRECTNESS`, or
`NO_GO_PERFORMANCE`; authentication, GPU interference, or transport failure is
`INCONCLUSIVE_ENVIRONMENT`.

- [ ] **Step 6: Update audit and handoff, then commit and push**

Record the exact tag, source commit/patch/tree hashes, verifier outcomes,
correctness boundary, controlled performance statistics, and any remaining
limitations. Stage only the audit and handoff, then commit with the required
trailer and push. Never stage `artifacts/`, `experiments/`, source archives,
PID files, or logs.
