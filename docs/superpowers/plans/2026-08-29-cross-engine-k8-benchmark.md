# Cross-Engine Exact-Greedy K8 Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. This repository's accepted workflow
> requires inline execution; do not use subagents or worktrees.

**Goal:** Build and execute a source-bound, storage-bounded benchmark that
compares TinyLLMForge Exact Greedy K8 with the strongest compatible public
vLLM greedy configuration on one admitted A100 using the same Qwen3-0.6B
workload.

**Architecture:** Keep all policy, schema, path, storage, metric, correctness,
and classification logic in dependency-light Python modules covered by local
tests. Run engine-specific code in isolated remote environments through one
worker protocol, while a local controller owns SSH, Kerberos, immutable
attempts, strict-clean GPU admission, process cleanup, and compact evidence
retrieval. Require producer, remote verifier, and frozen-source local verifier
agreement before assigning a cross-engine performance classification.

**Tech Stack:** Python 3.12, dataclasses, JSON/JSONL, hashlib, subprocess,
pytest, TinyLLMForge/PyTorch, pinned stable vLLM, NVIDIA NVML or `nvidia-smi`,
SSH, Git.

## Global Constraints

- The only authoritative checkout is `/Users/bytedance/Desktop/TinyLLMForge`,
  which resolves to `/Users/bytedance/dev/TinyLLMForge`.
- Do not create a worktree or use subagents.
- Keep and push only `origin/feat/kv-sparse-attention`.
- Stage exact task paths only; do not use broad `git add`, `git reset`,
  `git clean`, or mass formatting.
- Use `git -c core.hooksPath=/dev/null commit`.
- Every commit has exactly one
  `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Required SSH identity is `sitian@10.232.195.203`.
- Every SSH command must receive
  `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- Do not run `kinit`, `krenew`, or modify the user's Kerberos cache.
- Remote task data must stay below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/cross-engine-k8-qwen3-06b`.
- Do not write task data to remote `/`, remote `/tmp`, an old checkout, the
  Mac's `experiments/` tree, or the retired adaptive-ngram checkout.
- Reuse the read-only model at
  `/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B`; do not copy, hard-link,
  repack, or redownload it.
- Do not override `HOME`.
- Redirect `XDG_CACHE_HOME`, `HF_HOME`, `MODELSCOPE_CACHE`, `PIP_CACHE_DIR`,
  `UV_CACHE_DIR`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`,
  `CUDA_CACHE_PATH`, `PYTHONPYCACHEPREFIX`, and `TMPDIR` under the campaign
  root.
- Campaign-owned allocated bytes have a 16 GiB warning threshold and a
  20 GiB hard stop. The pre-existing model target is excluded.
- Local retained evidence must stay below
  `artifacts/cross_engine_k8/<run-tag>/`, use the exact allowlist from the
  design, and total no more than 50 MiB.
- Do not signal, terminate, reserve, or take over unrelated processes or
  GPUs. Cleanup is limited to the controller-recorded process group.
- Serialize every controller stage for the same host and attempt with a
  non-blocking local file lock so resumed sessions cannot concurrently mutate
  one remote environment or result bundle.
- Route vLLM ZeroMQ sockets through a short Linux abstract namespace via the
  public `VLLM_RPC_BASE_PATH` setting; do not create a short path outside the
  campaign root.
- On safe stage resumption, reuse only source-matching terminal worker output
  and preserve nonzero-exit sidecars under `failed-workers/` before retrying.
- GPU admission requires one A100 80GB PCIe, zero foreign compute processes,
  zero utilization at both admission samples, and no unauthorized memory
  occupancy.
- Freeze Qwen3-0.6B, BF16, TP1, batch 1, temperature 0, ignore EOS, prompt
  lengths 256/2048/8192, output length 128, two warmups, and seven measured
  repetitions.
- Required arms are `tinyllmforge_host_greedy`,
  `tinyllmforge_exact_k8`, and `vllm_default_greedy`.
- `vllm_public_multi_step` is conditional on a supported public control in
  the pinned stable release. Do not monkeypatch vLLM to create it.
- On the admitted driver `535.261.03`, cap candidate discovery at vLLM
  `0.11.2`, create its venv without system site-packages, and install the
  verified wheel with declared binary dependencies under campaign-scoped
  cache/temp paths. This is the highest retained pre-CUDA-13 candidate after
  host wheel-tag filtering.
- Thresholds are frozen before measured rows exist:
  TinyLLMForge K8 must improve aggregate median TPOT and throughput by at
  least 5%, have no bucket median-TPOT regression, keep TTFT/E2E/P95/P99/peak
  GPU memory/RSS regressions within 2%, and preserve exact output equality.
- Terminal classes are exactly `GO_CROSS_ENGINE_ADVANTAGE`,
  `CROSS_ENGINE_PARITY`, `NO_CROSS_ENGINE_ADVANTAGE`, and `INCOMPLETE`.

---

## File Map

### Dependency-light contracts

- Create `tools/cross_engine_k8_contract.py`: constants, approved-path checks,
  cache environment construction, storage accounting, immutable tags,
  Kerberos parsing, JSON helpers, schemas, and allowlist validation.
- Create `tools/cross_engine_k8_workload.py`: deterministic prompt-token
  generation, arm rotation, workload manifest, row identities, metric
  reconstruction, correctness reconciliation, aggregation, and terminal
  classification.
- Create `tools/cross_engine_k8_resources.py`: external process/GPU resource
  sampler, bounded sample file, and peak reduction.

### Engine execution

- Create `tools/cross_engine_k8_worker.py`: common worker CLI and result
  protocol plus TinyLLMForge host/K8 and vLLM default/public-multi-step
  adapters.
- Create `tools/cross_engine_k8_environment.py`: newest-first stable-vLLM
  candidate discovery, compatibility smoke protocol, environment manifest,
  source identity, and model inventory.

### Orchestration and verification

- Create `tools/run_cross_engine_k8_remote.py`: local controller, SSH and
  Kerberos checks, remote storage setup, atomic environment/source staging,
  strict-clean monitoring, worker launch, owned-process cleanup, smoke,
  canonical campaign, remote verifier launch, and allowlisted retrieval.
- Create `tools/verify_cross_engine_k8.py`: independent artifact parser,
  manifest/hash validation, matrix/correctness/metric recomputation, budget
  validation, and classification.
- Create `tools/stream_verify_cross_engine_k8_remote.py`: local frozen-source
  verifier that streams remote inputs one at a time and proves zero temporary
  retention.

### Tests and terminal documentation

- Create `tools/test_cross_engine_k8_contract.py`.
- Create `tools/test_cross_engine_k8_workload.py`.
- Create `tools/test_cross_engine_k8_resources.py`.
- Create `tools/test_cross_engine_k8_environment.py`.
- Create `tools/test_cross_engine_k8_worker.py`.
- Create `tools/test_run_cross_engine_k8_remote.py`.
- Create `tools/test_verify_cross_engine_k8.py`.
- Create `tools/test_stream_verify_cross_engine_k8_remote.py`.
- Create
  `docs/superpowers/audits/2026-08-29-cross-engine-k8-benchmark-audit.md`
  only after terminal evidence exists.
- Append a concise terminal reconciliation to `AGENT_HANDOFF_STATE.md` only
  after the audit is complete.

---

### Task 1: Approved paths, cache routing, storage budget, and Kerberos guard

**Files:**

- Create: `tools/cross_engine_k8_contract.py`
- Create: `tools/test_cross_engine_k8_contract.py`

**Interfaces:**

- Produces `CampaignPaths.create(remote_root: str, model_path: str)`.
- Produces `CampaignPaths.require_owned_remote(path: str) -> PurePosixPath`.
- Produces `cache_environment(paths: CampaignPaths) -> dict[str, str]`.
- Produces `classify_allocated_bytes(bytes_used: int) -> str`.
- Produces `parse_klist_lifetime(text: str, now: datetime) -> timedelta`.
- Produces `require_kerberos_coverage(lifetime, estimated, margin) -> None`.
- Produces `validate_attempt_tag(tag: str) -> str`.
- Produces `validate_local_allowlist(root: Path) -> dict`.

- [ ] **Step 1: Write failing boundary tests**

```python
def test_campaign_paths_reject_root_tmp_old_checkout_and_model_copy():
    paths = CampaignPaths.create(
        remote_root=REMOTE_ROOT,
        model_path=MODEL_PATH,
    )
    for forbidden in (
        "/",
        "/tmp/run",
        "/data00/home/sitian/TinyLLMForge/run",
        f"{REMOTE_ROOT}/shared/model-copy",
    ):
        with pytest.raises(ValueError):
            paths.require_owned_remote(forbidden)


def test_cache_environment_redirects_every_cache_without_home():
    env = cache_environment(
        CampaignPaths.create(
            remote_root=REMOTE_ROOT,
            model_path=MODEL_PATH,
        )
    )
    assert set(env) == set(REQUIRED_CACHE_VARIABLES)
    assert "HOME" not in env
    assert all(value.startswith(REMOTE_ROOT + "/") for value in env.values())


@pytest.mark.parametrize(
    ("bytes_used", "expected"),
    [
        (16 * GIB - 1, "OK"),
        (16 * GIB, "WARNING"),
        (20 * GIB - 1, "WARNING"),
        (20 * GIB, "HARD_STOP"),
    ],
)
def test_storage_boundaries(bytes_used, expected):
    assert classify_allocated_bytes(bytes_used) == expected


def test_kerberos_guard_requires_estimate_plus_thirty_minutes():
    with pytest.raises(RuntimeError, match="KERBEROS_TTL_INSUFFICIENT"):
        require_kerberos_coverage(
            lifetime=timedelta(hours=2),
            estimated=timedelta(hours=1, minutes=31),
            margin=timedelta(minutes=30),
        )
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_cross_engine_k8_contract.py -q
```

Expected: collection fails because `tools.cross_engine_k8_contract` does not
exist.

- [ ] **Step 3: Implement immutable constants and fail-closed validators**

Implement these exact public constants and result semantics:

```python
REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818/cross-engine-k8-qwen3-06b"
)
MODEL_PATH = "/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B"
WARNING_BYTES = 16 * 1024**3
HARD_STOP_BYTES = 20 * 1024**3
LOCAL_HARD_STOP_BYTES = 50 * 1024**2
REQUIRED_CACHE_VARIABLES = (
    "XDG_CACHE_HOME",
    "HF_HOME",
    "MODELSCOPE_CACHE",
    "PIP_CACHE_DIR",
    "UV_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "PYTHONPYCACHEPREFIX",
    "TMPDIR",
)
LOCAL_ALLOWLIST = (
    "controller_manifest.json",
    "environment_manifest.json",
    "workload_manifest.json",
    "case_rows.jsonl",
    "correctness_rows.jsonl",
    "comparison.json",
    "summary.json",
    "gate.json",
    "remote_verification.json",
    "local_verification.json",
    "manifest.sha256",
)
```

Use `PurePosixPath` lexical containment plus `resolve` output supplied by the
remote preflight; reject symlinks that escape `REMOTE_ROOT`. Parse `klist`
timestamps with the local timezone and require:

```python
lifetime >= estimated + margin
```

The attempt tag regex is:

```python
r"^20260829-cross-engine-k8-qwen3-06b-r[1-9][0-9]*$"
```

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_cross_engine_k8_contract.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit exact files**

```bash
git add \
  tools/cross_engine_k8_contract.py \
  tools/test_cross_engine_k8_contract.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(benchmark): add cross-engine campaign guards" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 2: Frozen workload, row identities, metric normalization, and gate

**Files:**

- Create: `tools/cross_engine_k8_workload.py`
- Create: `tools/test_cross_engine_k8_workload.py`

**Interfaces:**

- Consumes contract schema/version constants.
- Produces `build_workload_manifest(model_inventory_sha256: str) -> dict`.
- Produces `arm_order(repetition: int, eligible_arms: Sequence[str])`.
- Produces `expected_case_identities(manifest, eligible_arms) -> tuple`.
- Produces
  `reconstruct_metrics(request_start_ns: int, token_timestamps_ns:
  Sequence[int], request_end_ns: int, output_tokens: int) -> dict`.
- Produces `reconcile_correctness(rows, expected_tokens) -> dict`.
- Produces `aggregate_case_rows(rows) -> dict`.
- Produces `classify_comparison(comparison: dict) -> dict`.

- [ ] **Step 1: Write failing deterministic workload and threshold tests**

```python
def test_workload_is_frozen_and_prompt_tokens_are_deterministic():
    first = build_workload_manifest("a" * 64)
    second = build_workload_manifest("a" * 64)
    assert first == second
    assert first["prompt_lengths"] == [256, 2048, 8192]
    assert first["output_tokens"] == 128
    assert first["warmups"] == 2
    assert first["measured_repetitions"] == 7
    assert all(
        len(case["prompt_token_ids"]) == case["prompt_tokens"]
        for case in first["cases"]
    )


def test_rotation_balances_first_position():
    arms = REQUIRED_ARMS + ("vllm_public_multi_step",)
    orders = [arm_order(index, arms) for index in range(8)]
    assert {order[0] for order in orders} == set(arms)


def test_metric_reconstruction_uses_monotonic_token_timestamps():
    metrics = reconstruct_metrics(
        request_start_ns=0,
        token_timestamps_ns=[10, 20, 35, 55],
        request_end_ns=60,
        output_tokens=4,
    )
    assert metrics["ttft_ns"] == 10
    assert metrics["tpot_samples_ns"] == [10, 15, 20]
    assert metrics["e2e_ns"] == 60


def test_gate_requires_both_five_percent_gains_and_protected_metrics():
    comparison = passing_comparison_fixture()
    assert classify_comparison(comparison)["classification"] == (
        "GO_CROSS_ENGINE_ADVANTAGE"
    )
    comparison["aggregate"]["throughput_ratio"] = 1.049
    assert classify_comparison(comparison)["classification"] == (
        "NO_CROSS_ENGINE_ADVANTAGE"
    )
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_cross_engine_k8_workload.py -q
```

Expected: collection fails because `tools.cross_engine_k8_workload` does not
exist.

- [ ] **Step 3: Implement the frozen manifest and pure gate**

Use a fixed prompt seed, vocabulary-safe token range, and explicit BOS token
from the checkpoint metadata. Store prompt token IDs directly in the
manifest. Percentiles use nearest-rank. Aggregate each arm by first taking
the median within each context bucket, then the unweighted median across the
three buckets.

The strongest eligible vLLM arm is the one with the lower aggregate median
TPOT among correctness-valid vLLM arms. `classify_comparison` must return
`INCOMPLETE` before evaluating performance whenever required rows, exact
outputs, storage receipts, terminal receipts, or verifier agreement are
missing.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_cross_engine_k8_workload.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit exact files**

```bash
git add \
  tools/cross_engine_k8_workload.py \
  tools/test_cross_engine_k8_workload.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(benchmark): freeze cross-engine workload and gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 3: Environment discovery and pinned compatibility manifest

**Files:**

- Create: `tools/cross_engine_k8_environment.py`
- Create: `tools/test_cross_engine_k8_environment.py`

**Interfaces:**

- Consumes `CampaignPaths`, `cache_environment`, and workload manifest.
- Produces `StableRelease(version: str, requirement: str)`.
- Produces `candidate_releases(index_json: dict) -> tuple[StableRelease, ...]`.
- Produces `compatibility_decision(probes: Sequence[dict]) -> dict`.
- Produces
  `build_environment_manifest(tinyllmforge_probe: dict, vllm_probe: dict,
  model_inventory: dict, source_revision: str) -> dict`.
- Produces `build_model_inventory(model_root: Path) -> dict`.

- [ ] **Step 1: Write failing newest-first and fail-closed tests**

```python
def test_candidate_releases_are_stable_and_newest_first():
    releases = candidate_releases(pypi_fixture())
    assert [item.version for item in releases] == ["0.10.2", "0.10.1"]


def test_first_complete_compatible_release_is_frozen():
    decision = compatibility_decision([
        {"version": "0.10.2", "compatible": False, "reason": "driver"},
        {
            "version": "0.10.1",
            "compatible": True,
            "smoke_output_tokens": 128,
            "public_multi_step": False,
        },
    ])
    assert decision["selected_version"] == "0.10.1"
    assert decision["multi_step_status"] == (
        "VLLM_MULTI_STEP_NOT_PUBLICLY_AVAILABLE"
    )


def test_no_release_yields_incomplete_not_source_patch():
    decision = compatibility_decision([
        {"version": "0.10.2", "compatible": False, "reason": "model"},
    ])
    assert decision["classification"] == "INCOMPLETE_VLLM_COMPATIBILITY"
    assert decision["source_patch_allowed"] is False
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_cross_engine_k8_environment.py -q
```

Expected: collection fails because the environment module does not exist.

- [ ] **Step 3: Implement candidate and manifest logic**

The controller may obtain release metadata through `pip index versions vllm
--json` when supported, with a PyPI JSON fallback stored under the campaign
controller directory. Exclude prereleases and yanked releases. Probe
newest-first without modifying source. A successful probe records Python,
wheel hashes, `pip freeze`, PyTorch, CUDA runtime, Triton, FlashAttention,
driver, GPU UUID/name, model inventory hash, and public multi-step capability.

The model inventory walks files without following symlinks and stores only
relative path, allocated bytes, logical bytes, and SHA-256. It refuses any
inventory path outside the canonical shared checkpoint.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_cross_engine_k8_environment.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit exact files**

```bash
git add \
  tools/cross_engine_k8_environment.py \
  tools/test_cross_engine_k8_environment.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(benchmark): add pinned engine environment discovery" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 4: External GPU, process, and CPU resource sampler

**Files:**

- Create: `tools/cross_engine_k8_resources.py`
- Create: `tools/test_cross_engine_k8_resources.py`

**Interfaces:**

- Produces
  `ResourceSampler(pid, gpu_uuid, interval_s, max_samples, nvml_reader,
  process_reader, clock)`.
- Produces `ResourceSampler.sample() -> dict`.
- Produces `ResourceSampler.finalize() -> dict`.
- Produces `reduce_resource_samples(rows: Sequence[dict]) -> dict`.

- [ ] **Step 1: Write failing sampler and bounded-retention tests**

```python
def test_sampler_records_external_gpu_and_process_metrics():
    sampler = ResourceSampler(
        pid=123,
        gpu_uuid="GPU-a",
        interval_s=0.05,
        max_samples=3,
        nvml_reader=fake_nvml_reader,
        process_reader=fake_process_reader,
        clock=fake_clock,
    )
    row = sampler.sample()
    assert row["pid"] == 123
    assert row["gpu_uuid"] == "GPU-a"
    assert row["gpu_memory_bytes"] == 4_000
    assert row["rss_bytes"] == 8_000


def test_sampler_keeps_bounded_rows_and_exact_peaks():
    sampler = sampler_with_four_fixture_rows(max_samples=3)
    for _ in range(4):
        sampler.sample()
    final = sampler.finalize()
    assert final["samples_retained"] == 3
    assert final["samples_observed"] == 4
    assert final["peak_gpu_memory_bytes"] == 9_000
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_cross_engine_k8_resources.py -q
```

Expected: collection fails because the resource module does not exist.

- [ ] **Step 3: Implement an engine-independent sampler**

Prefer `pynvml` when installed; otherwise parse query-only `nvidia-smi` CSV.
Read RSS and CPU time from `/proc/<pid>/status` and `/proc/<pid>/stat`.
Maintain exact peaks independently of the bounded diagnostic ring. Mark
missing public metrics `NOT_EXPOSED`; never write zero for unavailable data.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_cross_engine_k8_resources.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit exact files**

```bash
git add \
  tools/cross_engine_k8_resources.py \
  tools/test_cross_engine_k8_resources.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(benchmark): add external engine resource sampler" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 5: Unified worker protocol and engine adapters

**Files:**

- Create: `tools/cross_engine_k8_worker.py`
- Create: `tools/test_cross_engine_k8_worker.py`
- Read/Reuse: `tools/profile_exact_greedy_decode_burst.py`

**Interfaces:**

- Consumes workload, environment, and resource contracts.
- Produces `EngineResult(token_ids, token_timestamps_ns, engine_metrics,
  retained_logits)`.
- Produces `TinyLLMForgeAdapter.run_case(case, arm) -> EngineResult`.
- Produces `VllmAdapter.run_case(case, arm) -> EngineResult`.
- Produces CLI:
  `python -m tools.cross_engine_k8_worker --plan PLAN --output DIR`.

- [ ] **Step 1: Write failing protocol and adapter contract tests**

```python
@pytest.mark.parametrize("arm", REQUIRED_ARMS)
def test_worker_emits_one_terminal_receipt_per_arm(tmp_path, arm):
    plan = worker_plan_fixture(tmp_path, arm=arm)
    result = run_worker(plan, adapters=fake_adapters())
    assert result["terminal"] is True
    assert result["arm"] == arm
    assert result["measured_rows"] == 3


def test_worker_rejects_engine_tokenizer_substitution(tmp_path):
    plan = worker_plan_fixture(tmp_path)
    plan["cases"][0].pop("prompt_token_ids")
    with pytest.raises(ValueError, match="prompt_token_ids"):
        run_worker(plan, adapters=fake_adapters())


def test_cross_engine_mismatch_excludes_performance(tmp_path):
    adapters = fake_adapters(vllm_tokens=[9] * 128)
    result = run_worker(worker_plan_fixture(tmp_path), adapters=adapters)
    assert result["correctness_valid"] is False
    assert result["performance_eligible"] is False
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_cross_engine_k8_worker.py -q
```

Expected: collection fails because the worker module does not exist.

- [ ] **Step 3: Implement the shared worker and TinyLLMForge adapter**

Reuse the existing profiler's prompt shape, Exact Greedy K8 activation,
counter collection, sampled-logit sidecars, and host-greedy path. Do not
modify runtime behavior. Each engine process performs two warmups, then all
three context buckets for exactly one repetition. Record monotonic request,
first-token, per-token, and end timestamps.

- [ ] **Step 4: Implement the vLLM public adapter**

Construct the engine with BF16, TP1, disabled prefix caching, greedy
temperature zero, ignore EOS, and max sequence length sufficient for
8192+128. Pass `prompt_token_ids` directly. Use only public stable APIs.
Detect a public multi-step control through inspected signatures/config
schemas, record the exact control and value, and omit the conditional arm
when absent.

- [ ] **Step 5: Run focused tests and confirm GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_cross_engine_k8_worker.py \
  tools/test_profile_exact_greedy_decode_burst.py -q
```

Expected: all tests pass and existing K8 profiler tests remain green.

- [ ] **Step 6: Commit exact files**

```bash
git add \
  tools/cross_engine_k8_worker.py \
  tools/test_cross_engine_k8_worker.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(benchmark): add TinyLLMForge and vLLM worker adapters" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 6: Independent producer/remote verifier

**Files:**

- Create: `tools/verify_cross_engine_k8.py`
- Create: `tools/test_verify_cross_engine_k8.py`

**Interfaces:**

- Consumes only immutable JSON/JSONL artifacts and source constants.
- Produces `verify_bundle(bundle_root: Path, expected_source: str) -> dict`.
- Produces CLI:
  `python -m tools.verify_cross_engine_k8 --bundle DIR --output FILE`.

- [ ] **Step 1: Write failing completeness and tamper tests**

```python
def test_verifier_recomputes_go_without_trusting_gate(tmp_path):
    bundle = write_valid_bundle(tmp_path, classification="INCOMPLETE")
    result = verify_bundle(bundle, expected_source="a" * 40)
    assert result["recomputed_classification"] == (
        "GO_CROSS_ENGINE_ADVANTAGE"
    )
    assert result["producer_agrees"] is False


def test_verifier_detects_case_row_tampering(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    append_text(bundle / "case_rows.jsonl", "{}\n")
    result = verify_bundle(bundle, expected_source="a" * 40)
    assert result["valid"] is False
    assert "MANIFEST_DIGEST_MISMATCH" in result["reasons"]


def test_verifier_marks_missing_terminal_receipt_incomplete(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    remove_terminal_receipt(bundle)
    result = verify_bundle(bundle, expected_source="a" * 40)
    assert result["recomputed_classification"] == "INCOMPLETE"
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_verify_cross_engine_k8.py -q
```

Expected: collection fails because the verifier does not exist.

- [ ] **Step 3: Implement independent recomputation**

The verifier must parse and validate every row, compare exact expected
identities, recompute token hashes and metrics from timestamps, reject
duplicates, validate source/environment/model/workload identities, recompute
storage usage and strongest-vLLM selection, and call the pure gate only after
correctness and completeness pass. It must not import producer/controller
functions other than immutable schema constants and pure workload math.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_verify_cross_engine_k8.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit exact files**

```bash
git add \
  tools/verify_cross_engine_k8.py \
  tools/test_verify_cross_engine_k8.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(benchmark): add independent cross-engine verifier" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 7: Local controller, immutable remote attempts, and owned cleanup

**Files:**

- Create: `tools/run_cross_engine_k8_remote.py`
- Create: `tools/test_run_cross_engine_k8_remote.py`

**Interfaces:**

- Consumes all prior contracts and worker/verifier CLIs.
- Produces `RemoteController(config, command_runner, clock)`.
- Produces `RemoteController.preflight() -> dict`.
- Produces `RemoteController.prepare_environments() -> dict`.
- Produces `RemoteController.wait_for_admitted_gpu() -> dict`.
- Produces `RemoteController.run_stage(stage: str) -> dict`.
- Produces `RemoteController.finalize() -> dict`.
- Produces CLI with
  `--stage preflight|prepare-environments|smoke|canonical|finalize`.

- [ ] **Step 1: Write failing SSH, admission, immutability, and cleanup tests**

```python
def test_controller_injects_file_cache_into_every_ssh_command():
    runner = RecordingRunner()
    controller = controller_fixture(runner=runner)
    controller.remote(["hostname"])
    assert runner.envs == [{
        "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian"
    }]


def test_controller_waits_for_two_strict_clean_samples():
    controller = controller_fixture(
        gpu_samples=[busy_gpu(), clean_gpu(), clean_gpu()]
    )
    admission = controller.wait_for_admitted_gpu()
    assert admission["sample_count"] == 3
    assert admission["admitted"] is True


def test_controller_never_kills_foreign_processes():
    controller = controller_fixture(
        owned_pgid=400,
        visible_processes=[{"pid": 99, "pgid": 99, "user": "other"}],
    )
    controller.cleanup_owned_processes()
    assert controller.signals_sent == [{"pgid": 400, "signal": "TERM"}]


def test_existing_attempt_directory_is_never_overwritten():
    controller = controller_fixture(remote_attempt_exists=True)
    with pytest.raises(RuntimeError, match="IMMUTABLE_ATTEMPT_EXISTS"):
        controller.run_stage("smoke")
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_run_cross_engine_k8_remote.py -q
```

Expected: collection fails because the controller does not exist.

- [ ] **Step 3: Implement preflight and atomic environment preparation**

Preflight runs local `klist`, TTL coverage, SSH identity, `df` for `/` and
`/data00/home/sitian`, canonical path checks, model inventory, campaign
allocated-byte accounting, and existing environment discovery. Build each
environment in `<name>.building-<tag>` and rename only after compatibility
and manifest completion. Source staging uses an exact tracked-file archive
plus recorded dirty-state refusal for runtime files.

- [ ] **Step 4: Implement monitor, worker launch, and owned cleanup**

Admission polls remotely but decisions remain in the current local agent.
Require two clean samples separated by at least five seconds. Launch each
worker with `setsid`, write PGID/PID/command/source receipts atomically, and
poll bounded heartbeat/status files. On timeout or controller interruption,
signal only the recorded owned PGID, verify exit, and retain a cleanup
receipt.

- [ ] **Step 5: Implement smoke, canonical rotation, and finalization**

Smoke uses one context, one measured repetition, and all eligible arms; mark
it non-performance evidence. Canonical uses fresh engine processes per arm
and repetition, seven repetitions, deterministic balanced arm rotation, and
storage/admission checks before every worker. Finalization assembles the
allowlisted bundle, writes `manifest.sha256`, invokes the remote verifier,
and writes an immutable terminal receipt.

- [ ] **Step 6: Run focused tests and confirm GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_run_cross_engine_k8_remote.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit exact files**

```bash
git add \
  tools/run_cross_engine_k8_remote.py \
  tools/test_run_cross_engine_k8_remote.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(benchmark): add remote cross-engine controller" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 8: Frozen-source streaming local verifier and 50 MiB retention gate

**Files:**

- Create: `tools/stream_verify_cross_engine_k8_remote.py`
- Create: `tools/test_stream_verify_cross_engine_k8_remote.py`

**Interfaces:**

- Consumes remote manifest and one remote file stream at a time.
- Produces `stream_verify(config, ssh_runner, verifier) -> dict`.
- Produces CLI with `--remote-run-tag`, `--local-root`, and
  `--expected-source`.

- [ ] **Step 1: Write failing allowlist, size, and cleanup tests**

```python
def test_streaming_verifier_does_not_retain_non_allowlisted_inputs(tmp_path):
    remote = remote_fixture(extra_file="worker.log")
    result = stream_verify(
        config_fixture(tmp_path),
        remote,
        fake_verifier,
    )
    assert result["valid"] is True
    assert list(tmp_path.rglob("worker.log")) == []


def test_streaming_verifier_rejects_fifty_mib_boundary(tmp_path):
    remote = remote_fixture(total_bytes=50 * 1024**2 + 1)
    with pytest.raises(RuntimeError, match="LOCAL_STORAGE_HARD_STOP"):
        stream_verify(config_fixture(tmp_path), remote, fake_verifier)


def test_streaming_verifier_removes_temporary_files_on_failure(tmp_path):
    remote = remote_fixture(corrupt_after="case_rows.jsonl")
    with pytest.raises(RuntimeError):
        stream_verify(config_fixture(tmp_path), remote, fake_verifier)
    assert list(tmp_path.rglob("*.streaming-tmp")) == []
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_stream_verify_cross_engine_k8_remote.py -q
```

Expected: collection fails because the streaming verifier does not exist.

- [ ] **Step 3: Implement one-file streaming and atomic retained output**

Use a task-specific `mkstemp` path inside the target run directory, never
`/tmp`. For each remote file: stream, hash, parse or copy only when
allowlisted, then unlink in `finally`. Write retained files with
`os.replace`. Run the frozen-source local verifier from the authoritative
checkout and require equality with the remote verifier's matrix,
classification, source, model, workload, environment, and manifest digest.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_stream_verify_cross_engine_k8_remote.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit exact files**

```bash
git add \
  tools/stream_verify_cross_engine_k8_remote.py \
  tools/test_stream_verify_cross_engine_k8_remote.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(benchmark): add streaming local evidence verifier" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 9: Full local verification and source freeze

**Files:**

- Verify all files from Tasks 1-8.

**Interfaces:**

- Produces one source commit used by both remote source staging and local
  frozen-source verification.

- [ ] **Step 1: Run the complete focused test matrix**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_cross_engine_k8_contract.py \
  tools/test_cross_engine_k8_workload.py \
  tools/test_cross_engine_k8_environment.py \
  tools/test_cross_engine_k8_resources.py \
  tools/test_cross_engine_k8_worker.py \
  tools/test_run_cross_engine_k8_remote.py \
  tools/test_verify_cross_engine_k8.py \
  tools/test_stream_verify_cross_engine_k8_remote.py \
  tools/test_profile_exact_greedy_decode_burst.py \
  tools/test_exact_greedy_decode_burst_verify.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run static and diff checks**

Run:

```bash
/opt/homebrew/bin/python3.12 -m compileall -q \
  tools/cross_engine_k8_contract.py \
  tools/cross_engine_k8_workload.py \
  tools/cross_engine_k8_environment.py \
  tools/cross_engine_k8_resources.py \
  tools/cross_engine_k8_worker.py \
  tools/run_cross_engine_k8_remote.py \
  tools/verify_cross_engine_k8.py \
  tools/stream_verify_cross_engine_k8_remote.py
git diff --check
git status --short -- \
  tools/cross_engine_k8_contract.py \
  tools/cross_engine_k8_workload.py \
  tools/cross_engine_k8_environment.py \
  tools/cross_engine_k8_resources.py \
  tools/cross_engine_k8_worker.py \
  tools/run_cross_engine_k8_remote.py \
  tools/verify_cross_engine_k8.py \
  tools/stream_verify_cross_engine_k8_remote.py
```

Expected: compile and diff checks exit zero; task paths have no unstaged
changes.

- [ ] **Step 3: Record and push the frozen source identity**

Run:

```bash
git rev-parse HEAD
git push origin feat/kv-sparse-attention
git ls-remote origin refs/heads/feat/kv-sparse-attention
```

Expected: local HEAD, tracking branch, and remote branch SHA are identical.

### Task 10: Remote storage/environment preflight

**Files:**

- Remote only under the approved campaign root.
- Local compact controller receipts only after finalization.

**Interfaces:**

- Consumes the frozen source SHA from Task 9.
- Produces remote storage, model, source, environment, and compatibility
  manifests without a canonical GPU run.

- [ ] **Step 1: Verify the active FILE cache and estimated coverage**

Run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
  klist
/opt/homebrew/bin/python3.12 \
  tools/run_cross_engine_k8_remote.py \
  --stage preflight \
  --host sitian@10.232.195.203 \
  --run-tag 20260829-cross-engine-k8-qwen3-06b-r1
```

Expected: the controller either writes a complete preflight receipt or exits
with a specific `INCOMPLETE_*` reason before mutating remote state.

- [ ] **Step 2: Prepare isolated environments under the hard budget**

Run:

```bash
/opt/homebrew/bin/python3.12 \
  tools/run_cross_engine_k8_remote.py \
  --stage prepare-environments \
  --host sitian@10.232.195.203 \
  --run-tag 20260829-cross-engine-k8-qwen3-06b-r1
```

Expected: TinyLLMForge and the first newest-first compatible stable vLLM
environment are atomically finalized, campaign allocated bytes remain below
20 GiB, and no model copy appears under the campaign root.

- [ ] **Step 3: Inspect exact remote receipts**

Run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
  ssh sitian@10.232.195.203 \
  "python3 -m json.tool \
  /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/\
cross-engine-k8-qwen3-06b/attempts/\
20260829-cross-engine-k8-qwen3-06b-r1/controller/preflight.json"
```

Expected: canonical model path, source SHA, cache routing, root/data00 `df`,
allocated campaign bytes, selected vLLM version, and compatibility decision
are explicit.

### Task 11: Strict-clean smoke

**Files:**

- Remote smoke attempt under the approved campaign root.

**Interfaces:**

- Produces non-performance smoke receipts for every eligible arm.

- [ ] **Step 1: Start local-agent-owned admission monitor and smoke**

Run:

```bash
/opt/homebrew/bin/python3.12 \
  tools/run_cross_engine_k8_remote.py \
  --stage smoke \
  --host sitian@10.232.195.203 \
  --run-tag 20260829-cross-engine-k8-qwen3-06b-r1
```

Expected: the local controller waits until two strict-clean samples, launches
all eligible arms, records exact 128-token outputs, and never terminates
foreign work.

- [ ] **Step 2: Verify smoke correctness and resource receipts**

Run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
  ssh sitian@10.232.195.203 \
  "python3 -m tools.verify_cross_engine_k8 \
  --bundle /data00/home/sitian/tinyllmforge-workspaces/\
command-timeline-20260818/cross-engine-k8-qwen3-06b/attempts/\
20260829-cross-engine-k8-qwen3-06b-r1/smoke \
  --mode smoke"
```

Expected: all required eligible arms pass exact-output and length checks;
smoke remains explicitly excluded from performance classification.

### Task 12: Canonical campaign, dual verification, and compact retrieval

**Files:**

- Remote canonical attempt under the approved campaign root.
- Create locally only:
  `artifacts/cross_engine_k8/20260829-cross-engine-k8-qwen3-06b-r1/`
  with the exact allowlist.

**Interfaces:**

- Produces the only eligible horizontal performance verdict.

- [ ] **Step 1: Run the complete frozen canonical matrix**

Run:

```bash
/opt/homebrew/bin/python3.12 \
  tools/run_cross_engine_k8_remote.py \
  --stage canonical \
  --host sitian@10.232.195.203 \
  --run-tag 20260829-cross-engine-k8-qwen3-06b-r1
```

Expected: 7 repetitions × 3 contexts for every eligible arm, exact worker
terminal receipts, storage below 20 GiB, and no source or threshold drift.

- [ ] **Step 2: Finalize and run the remote independent verifier**

Run:

```bash
/opt/homebrew/bin/python3.12 \
  tools/run_cross_engine_k8_remote.py \
  --stage finalize \
  --host sitian@10.232.195.203 \
  --run-tag 20260829-cross-engine-k8-qwen3-06b-r1
```

Expected: immutable remote final bundle, manifest, producer gate, remote
verification, and terminal receipt all exist.

- [ ] **Step 3: Stream the compact allowlist and run local verification**

Run:

```bash
/opt/homebrew/bin/python3.12 \
  tools/stream_verify_cross_engine_k8_remote.py \
  --host sitian@10.232.195.203 \
  --remote-run-tag 20260829-cross-engine-k8-qwen3-06b-r1 \
  --local-root artifacts/cross_engine_k8 \
  --expected-source "$(git rev-parse HEAD)"
```

Expected: local retained bytes are at most 50 MiB, no streaming temporary
files remain, and producer/remote/local classifications and hashes agree.

### Task 13: Audit, handoff reconciliation, final verification, and push

**Files:**

- Create:
  `docs/superpowers/audits/2026-08-29-cross-engine-k8-benchmark-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Add exact allowlisted evidence files under:
  `artifacts/cross_engine_k8/20260829-cross-engine-k8-qwen3-06b-r1/`

**Interfaces:**

- Produces a claim-bounded, source- and evidence-linked terminal record.

- [ ] **Step 1: Write the audit from verified artifacts**

The audit must report:

```text
source SHA and remote SHA
pinned vLLM version and public multi-step availability
GPU/model/workload/environment identity
within-TinyLLMForge K8 versus host result
within-vLLM multi-step versus default result, when eligible
absolute TinyLLMForge K8 versus strongest eligible vLLM result
TTFT, median/P95/P99 TPOT, E2E, throughput, GPU memory, RSS
correctness result
storage cost and local retention cost
producer, remote verifier, and local verifier agreement
terminal classification
claim boundary and all incomplete/negative evidence
```

- [ ] **Step 2: Append a concise EOF reconciliation**

Append to `AGENT_HANDOFF_STATE.md` the run tag, source SHA, selected vLLM
version, artifact path, classification, key benefit/cost numbers, and exact
next action. Do not rewrite historical sections.

- [ ] **Step 3: Run fresh terminal verification**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_cross_engine_k8_contract.py \
  tools/test_cross_engine_k8_workload.py \
  tools/test_cross_engine_k8_environment.py \
  tools/test_cross_engine_k8_resources.py \
  tools/test_cross_engine_k8_worker.py \
  tools/test_run_cross_engine_k8_remote.py \
  tools/test_verify_cross_engine_k8.py \
  tools/test_stream_verify_cross_engine_k8_remote.py \
  tools/test_profile_exact_greedy_decode_burst.py \
  tools/test_exact_greedy_decode_burst_verify.py -q
python3 -m json.tool \
  artifacts/cross_engine_k8/20260829-cross-engine-k8-qwen3-06b-r1/gate.json \
  >/dev/null
python3 -m json.tool \
  artifacts/cross_engine_k8/20260829-cross-engine-k8-qwen3-06b-r1/\
remote_verification.json >/dev/null
python3 -m json.tool \
  artifacts/cross_engine_k8/20260829-cross-engine-k8-qwen3-06b-r1/\
local_verification.json >/dev/null
git diff --check
```

Expected: all tests and JSON checks pass; diff check exits zero.

- [ ] **Step 4: Stage only terminal task files and inspect the staged diff**

Run:

```bash
git add \
  docs/superpowers/audits/2026-08-29-cross-engine-k8-benchmark-audit.md \
  AGENT_HANDOFF_STATE.md \
  artifacts/cross_engine_k8/20260829-cross-engine-k8-qwen3-06b-r1/\
controller_manifest.json \
  artifacts/cross_engine_k8/20260829-cross-engine-k8-qwen3-06b-r1/\
environment_manifest.json \
  artifacts/cross_engine_k8/20260829-cross-engine-k8-qwen3-06b-r1/\
workload_manifest.json \
  artifacts/cross_engine_k8/20260829-cross-engine-k8-qwen3-06b-r1/\
case_rows.jsonl \
  artifacts/cross_engine_k8/20260829-cross-engine-k8-qwen3-06b-r1/\
correctness_rows.jsonl \
  artifacts/cross_engine_k8/20260829-cross-engine-k8-qwen3-06b-r1/\
comparison.json \
  artifacts/cross_engine_k8/20260829-cross-engine-k8-qwen3-06b-r1/summary.json \
  artifacts/cross_engine_k8/20260829-cross-engine-k8-qwen3-06b-r1/gate.json \
  artifacts/cross_engine_k8/20260829-cross-engine-k8-qwen3-06b-r1/\
remote_verification.json \
  artifacts/cross_engine_k8/20260829-cross-engine-k8-qwen3-06b-r1/\
local_verification.json \
  artifacts/cross_engine_k8/20260829-cross-engine-k8-qwen3-06b-r1/\
manifest.sha256
git diff --cached --check
git diff --cached --stat
```

Expected: only the audit, EOF handoff append, and exact compact allowlist are
staged.

- [ ] **Step 5: Commit, push, and verify the remote SHA**

Run:

```bash
git -c core.hooksPath=/dev/null commit \
  -m "perf(benchmark): compare Exact K8 with vLLM" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
git rev-parse HEAD
git rev-parse origin/feat/kv-sparse-attention
git ls-remote origin refs/heads/feat/kv-sparse-attention
```

Expected: local HEAD, tracking SHA, and GitHub branch SHA are identical.
