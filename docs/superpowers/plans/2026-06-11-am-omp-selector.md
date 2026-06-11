# Attention Matching OMP Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable OMP-style key selector to the existing Attention Matching compact decode path so TinyLLMForge can compare AM-HighestAttnKeys against AM-OMP on local tests and real needle runs.

**Architecture:** Keep this as a minimal quality-validation path: OMP only replaces key-index selection, while the existing beta box fitting and compact-value least-squares fitting remain unchanged. The decode integration adds a selector config and routes both fp16-KV and KV8 AM paths through the same `attention_matching_decode(...)` interface. OMP searches a small HighestAttnKeys prefilter pool instead of every historical token so long-context smoke runs remain usable; no caching, low-frequency refresh, or persistent execution-shape optimization is introduced in this plan.

**Tech Stack:** Python, PyTorch, TinyLLMForge eager Attention Matching utilities, dependency-light script tests.

---

## File Structure

- Modify `tinyvllm/engine/attention_matching.py`: add OMP selector primitives, add `selector` arguments, and dispatch between HighestAttnKeys and OMP.
- Modify `tinyvllm/config.py`: add `am_compact_selector` with validation.
- Modify `tinyvllm/utils/context.py`: carry the selector through the global attention context.
- Modify `tinyvllm/engine/model_runner.py`: pass config selector into attention context.
- Modify `tinyvllm/layers/attention.py`: pass selector into `attention_matching_decode(...)` in both KV8 and fp16 paths.
- Modify `tools/test_attention_matching.py`: add failing tests for OMP selector and decode dispatch.
- Modify `tools/eval_needle.py`: expose `--am-compact-selector` so remote quality smoke can compare `highest` and `omp`.
- Modify `docs/qwen3-8b-fixes.md`: record the new OMP selector implementation scope and verification results.

## Task 1: Add OMP selector unit tests

**Files:**
- Modify: `tools/test_attention_matching.py`

- [ ] **Step 1: Import the new symbols in the test file**

Change the import block in `tools/test_attention_matching.py` to include the OMP selector and generic compact function:

```python
from tinyvllm.engine.attention_matching import (  # noqa: E402
    attention_matching_compact_keys,
    attention_matching_highest_keys,
    attention_matching_decode,
    attention_output,
    fit_attention_bias,
    fit_compacted_values,
    highest_attention_key_indices,
    omp_attention_key_indices,
)
```

- [ ] **Step 2: Add failing selector shape and uniqueness test**

Append this test after `test_highest_attention_key_indices_selects_dominant_key_by_rms_attention`:

```python
def test_omp_attention_key_indices_returns_budget_unique_indices():
    torch.manual_seed(10)
    queries = torch.randn(5, 4)
    keys = torch.randn(9, 4)
    values = torch.randn(9, 4)

    indices = omp_attention_key_indices(keys, values, queries, budget=4, ridge_lambda=1e-6)

    assert indices.shape == (4,)
    assert indices.dtype == torch.long
    assert len(set(indices.tolist())) == 4
    assert torch.all(indices >= 0)
    assert torch.all(indices < keys.shape[0])
```

- [ ] **Step 3: Add failing synthetic quality test**

Append this test after the uniqueness test:

```python
def test_omp_selector_reduces_error_vs_highest_keys_on_synthetic_case():
    queries = torch.tensor([[6.0, 0.0], [0.0, 6.0]], dtype=torch.float32)
    keys = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.8],
        [-1.0, 0.0],
    ], dtype=torch.float32)
    values = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
        [-1.0, 0.0],
    ], dtype=torch.float32)
    target = attention_output(queries, keys, values)

    highest = attention_matching_compact_keys(keys, values, queries, budget=2, selector="highest")
    omp = attention_matching_compact_keys(keys, values, queries, budget=2, selector="omp")

    highest_out = attention_output(queries, highest.keys, highest.values, highest.beta)
    omp_out = attention_output(queries, omp.keys, omp.values, omp.beta)

    highest_mse = torch.mean((highest_out - target) ** 2)
    omp_mse = torch.mean((omp_out - target) ** 2)

    assert omp_mse <= highest_mse + 1e-6
```

- [ ] **Step 4: Add failing decode dispatch test**

Append this test after `test_attention_matching_decode_supports_gqa_and_compact_output_shape`:

```python
def test_attention_matching_decode_accepts_omp_selector():
    torch.manual_seed(11)
    q = torch.randn(1, 4, 6)
    keys = torch.randn(1, 12, 2, 6)
    values = torch.randn(1, 12, 2, 6)
    context_lens = torch.tensor([12], dtype=torch.int32)

    out = attention_matching_decode(q, keys, values, context_lens, budget=4, selector="omp")

    assert out.shape == (1, 4, 6)
    assert out.dtype == q.dtype
```

- [ ] **Step 5: Register the tests in `main()`**

Update `main()` in `tools/test_attention_matching.py`:

```python
def main():
    test_highest_attention_key_indices_selects_dominant_key_by_rms_attention()
    test_omp_attention_key_indices_returns_budget_unique_indices()
    test_omp_selector_reduces_error_vs_highest_keys_on_synthetic_case()
    test_fit_attention_bias_preserves_attention_mass_for_selected_keys()
    test_fit_compacted_values_reduces_attention_output_error_vs_direct_values()
    test_attention_matching_highest_keys_returns_compacted_cache_and_indices()
    test_attention_matching_decode_supports_gqa_and_compact_output_shape()
    test_attention_matching_decode_accepts_omp_selector()
    test_attention_matching_decode_matches_full_attention_when_budget_covers_cache()
    print("attention matching tests passed")
```

- [ ] **Step 6: Run test to verify RED**

Run:

```bash
python3 tools/test_attention_matching.py
```

Expected: fail with `ImportError` for `attention_matching_compact_keys` or `omp_attention_key_indices`.

## Task 2: Implement the OMP selector primitive

**Files:**
- Modify: `tinyvllm/engine/attention_matching.py`
- Test: `tools/test_attention_matching.py`

- [ ] **Step 1: Add a helper for fitted-output error**

Insert this helper after `highest_attention_key_indices(...)`:

```python
def _fit_selected_output_error(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    selected_indices: torch.Tensor,
    target: torch.Tensor,
    ridge_lambda: float,
) -> torch.Tensor:
    if selected_indices.numel() == 0:
        return torch.mean(target * target)
    beta = fit_attention_bias(keys, queries, selected_indices, beta_bound=3.0)
    compact_values = fit_compacted_values(
        keys,
        values,
        queries,
        selected_indices,
        beta,
        ridge_lambda=ridge_lambda,
    )
    pred = attention_output(queries, keys[selected_indices], compact_values, beta)
    return torch.mean((pred - target) ** 2)
```

- [ ] **Step 2: Add `omp_attention_key_indices(...)`**

Insert this function after `_fit_selected_output_error(...)`:

```python
def omp_attention_key_indices(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    budget: int,
    ridge_lambda: float = 1e-6,
    beta_bound: float = 3.0,
    score_method: str = "rms",
    candidate_pool_size: int | None = None,
) -> torch.Tensor:
    """Select compact key indices by greedy output-matching OMP.

    The objective is the same attention-output MSE used by the AM value-fitting
    stage. To keep long-context smoke runs usable, OMP only searches a small
    HighestAttnKeys candidate pool rather than every historical token. At each
    iteration we try every unselected candidate, refit beta/C_v for the temporary
    set, and keep the candidate with the lowest fitted-output error.
    """
    num_keys = keys.shape[0]
    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget >= num_keys:
        return torch.arange(num_keys, device=keys.device, dtype=torch.long)

    target = attention_output(queries, keys, values)
    selected: list[int] = []
    if candidate_pool_size is None or candidate_pool_size <= 0:
        candidate_pool_size = max(budget + 4, budget * 2)
    candidate_pool_size = max(budget, min(num_keys, candidate_pool_size))
    pool = highest_attention_key_indices(
        keys,
        queries,
        budget=candidate_pool_size,
        score_method=score_method,
    )
    remaining = pool.tolist()
    for _ in range(budget):
        best_candidate = remaining[0]
        best_error: torch.Tensor | None = None
        for candidate in remaining:
            trial = torch.tensor(selected + [candidate], device=keys.device, dtype=torch.long)
            err = _fit_selected_output_error(keys, values, queries, trial, target, ridge_lambda)
            if best_error is None or float(err.item()) < float(best_error.item()):
                best_error = err
                best_candidate = candidate
        selected.append(best_candidate)
        remaining.remove(best_candidate)
    return torch.tensor(selected, device=keys.device, dtype=torch.long)
```

- [ ] **Step 3: Add generic compact function with selector dispatch**

Insert this function before `attention_matching_highest_keys(...)`:

```python
def attention_matching_compact_keys(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    budget: int,
    selector: str = "highest",
    score_method: str = "rms",
    beta_bound: float = 3.0,
    nnls_iters: int = 0,
    ridge_lambda: float = 0.0,
) -> AttentionMatchedKV:
    """Run Attention Matching with a configurable key selector."""
    if selector == "highest":
        selected = highest_attention_key_indices(keys, queries, budget, score_method)
    elif selector == "omp":
        selected = omp_attention_key_indices(keys, values, queries, budget, ridge_lambda)
    else:
        raise ValueError("selector must be 'highest' or 'omp'")
    beta = fit_attention_bias(keys, queries, selected, beta_bound, nnls_iters)
    compact_values = fit_compacted_values(keys, values, queries, selected, beta, ridge_lambda)
    return AttentionMatchedKV(
        keys=keys[selected].contiguous(),
        beta=beta.contiguous(),
        values=compact_values.contiguous(),
        indices=selected.contiguous(),
    )
```

- [ ] **Step 4: Reuse generic compact function from `attention_matching_highest_keys(...)`**

Replace the body of `attention_matching_highest_keys(...)` with:

```python
    """Run AM-HighestAttnKeys for one layer/head."""
    return attention_matching_compact_keys(
        keys,
        values,
        queries,
        budget=budget,
        selector="highest",
        score_method=score_method,
        beta_bound=beta_bound,
        nnls_iters=nnls_iters,
        ridge_lambda=ridge_lambda,
    )
```

- [ ] **Step 5: Run selector tests to verify GREEN for primitive tests**

Run:

```bash
python3 tools/test_attention_matching.py
```

Expected: still fail at decode selector test because `attention_matching_decode(...)` does not accept `selector` yet.

## Task 3: Route selector through decode and config

**Files:**
- Modify: `tinyvllm/engine/attention_matching.py`
- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/utils/context.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/layers/attention.py`
- Test: `tools/test_attention_matching.py`

- [ ] **Step 1: Add selector argument to `attention_matching_decode(...)`**

Change the signature in `tinyvllm/engine/attention_matching.py` to:

```python
def attention_matching_decode(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    context_lens: torch.Tensor,
    budget: int,
    selector: str = "highest",
    score_method: str = "rms",
    beta_bound: float = 3.0,
    ridge_lambda: float = 1e-6,
) -> torch.Tensor:
```

- [ ] **Step 2: Dispatch via generic compact function in decode**

Replace the compact call inside `attention_matching_decode(...)` with:

```python
                compact = attention_matching_compact_keys(
                    k_seq,
                    v_seq,
                    q_group,
                    budget=budget,
                    selector=selector,
                    score_method=score_method,
                    beta_bound=beta_bound,
                    ridge_lambda=ridge_lambda,
                )
```

- [ ] **Step 3: Add config field and validation**

In `tinyvllm/config.py`, add this field after `am_compact_blocks`:

```python
    am_compact_selector: str = "highest"                  # highest / omp：AM compact key 选择器
```

Then add this validation after `assert self.am_compact_min_seq_len >= 0`:

```python
        assert self.am_compact_selector in ("highest", "omp"), \
            "am_compact_selector 仅支持 highest / omp"
```

- [ ] **Step 4: Add selector to context defaults**

In `tinyvllm/utils/context.py`, add the selector field to the context dataclass or context object next to existing AM fields:

```python
    am_compact_selector: str = "highest"
```

- [ ] **Step 5: Pass selector from model runner into context**

In `tinyvllm/engine/model_runner.py`, in the place that builds/sets `set_context(...)`, add:

```python
            am_compact_selector=self.config.am_compact_selector,
```

next to `am_compact_blocks`, `am_compact_score_method`, `am_compact_beta_bound`, and `am_compact_ridge_lambda`.

- [ ] **Step 6: Pass selector from attention layer to decode in both AM branches**

In `tinyvllm/layers/attention.py`, add this argument to both `attention_matching_decode(...)` calls:

```python
                        selector=context.am_compact_selector,
```

Place it immediately after `budget=context.am_compact_blocks`.

- [ ] **Step 7: Run attention matching tests**

Run:

```bash
python3 tools/test_attention_matching.py
```

Expected: `attention matching tests passed`.

## Task 4: Expose selector in eval CLI and document scope

**Files:**
- Modify: `tools/eval_needle.py`
- Modify: `docs/qwen3-8b-fixes.md`

- [ ] **Step 1: Add CLI flag to eval tool**

In `tools/eval_needle.py`, add this argument next to existing AM flags:

```python
    p.add_argument("--am-compact-selector", type=str, default="highest", choices=["highest", "omp"],
                   help="Attention Matching compact decode 的 key selector：highest 或 omp。")
```

- [ ] **Step 2: Include selector in label**

Change the AM label construction from:

```python
        label = f"AM-HighestAttnKeys b={args.am_compact_blocks}"
```

to:

```python
        am_name = "AM-HighestAttnKeys" if args.am_compact_selector == "highest" else "AM-OMP"
        label = f"{am_name} b={args.am_compact_blocks}"
```

- [ ] **Step 3: Pass selector into LLM config kwargs**

In the config dict or constructor kwargs that already pass `am_compact_blocks`, add:

```python
        "am_compact_selector": args.am_compact_selector,
```

- [ ] **Step 4: Add documentation subsection**

Append this section after §45.5 in `docs/qwen3-8b-fixes.md`:

```markdown

### 45.6 AM-OMP selector 接入（2026-06-11）

承接 §45.5：AM-HighestAttnKeys b=16/32 已达到 Quest top-k=16 的质量门槛，因此进入 OMP-fast 方向。
本轮先做最小质量验证入口，而不是吞吐优化版：只把 Attention Matching 的 key selector 从
`highest` 扩展为可选 `omp`，beta box fitting 与 `C_v` least-squares 仍复用 §45.2 的实现。

实现范围：

- `tinyvllm/engine/attention_matching.py`：新增 `omp_attention_key_indices()` 与
  `attention_matching_compact_keys(selector=...)`；
- `Config.am_compact_selector`：支持 `highest` / `omp`；
- `tools/eval_needle.py --am-compact-selector`：真实 needle 评测可直接切换 selector。

本轮刻意不做：

- compact tensor cache；
- decode step 低频 refresh；
- block/layer 级 selector 结果复用；
- OMP GPU kernel 化。

原因是先分离质量问题和执行形态问题：若 OMP 在 fixed-prompt needle 上不能优于 HighestAttnKeys，
则不值得继续优化工程路径；若 OMP 质量更稳，再进入 OMP-fast execution-shape 优化。

本地验证：

```bash
python3 -m py_compile tinyvllm/engine/attention_matching.py tinyvllm/config.py tinyvllm/utils/context.py tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/eval_needle.py tools/test_attention_matching.py
python3 tools/test_attention_matching.py
```
```

- [ ] **Step 5: Run compile and tests**

Run:

```bash
python3 -m py_compile tinyvllm/engine/attention_matching.py tinyvllm/config.py tinyvllm/utils/context.py tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/eval_needle.py tools/test_attention_matching.py
python3 tools/test_attention_matching.py
```

Expected: py_compile succeeds and prints `attention matching tests passed`.

## Self-Review

- Spec coverage: plan covers selector tests, OMP primitive, decode/config propagation, eval CLI exposure, and documentation.
- Placeholder scan: no TBD/TODO placeholders remain; all code steps include exact snippets or commands.
- Type consistency: selector values are consistently `"highest"` and `"omp"`; new public functions are consistently named `omp_attention_key_indices` and `attention_matching_compact_keys`.
- Scope check: plan intentionally excludes cache/refresh/execution-shape optimization so the first OMP step remains a small, testable quality-validation path.
