# DFlash Feasibility Spike

本文只做 DFlash / diffusion-style block drafting 接入可行性预研，不直接实现完整 runtime。

## 结论

建议先做 toy / interface spike，不建议马上在 TinyLLMForge 里完整实现 DFlash。

原因：

1. 当前 KV offload blockwise prefill/decode 主线已经进入 correctness + smoke 阶段，DFlash 是另一条 speculative architecture 线，直接实现会把风险混在一起。
2. DFlash 需要 draft diffusion model、target hidden state 暴露、block-level draft sampling、target verify/commit pipeline、checkpoint/权重加载约定；不是一个 attention kernel 小改动。
3. TinyLLMForge 已有 n-gram speculative profiler 和 target verify/commit 原语，可以先抽象共同接口，再决定是否接完整 DFlash。

## DFlash 关键机制理解

DFlash 的核心不是 KV offload，也不是 FlashAttention 替代品，而是 block-level speculative decoding：

- draft 侧不是逐 token 自回归，而是用 block diffusion / denoising 思路一次提出一段 draft tokens；
- target model 仍负责验证，接受 draft prefix，并在 mismatch 后回到普通生成；
- 目标收益来自减少 target autoregressive steps，而不是减少单次 attention 的数学复杂度；
- 正确性要求与 speculative decoding 一致：最终分布/输出不能被 draft model 破坏。

对 TinyLLMForge 来说，DFlash 更接近 `tools/profile_ngram_commit.py` 的 speculative verify/commit 实验线，而不是 `tinyvllm/layers/attention.py` 的 blockwise prefill/decode correctness path。

## 当前项目可复用部件

### 已有 speculative helper

- `tinyvllm/speculative/ngram.py`
  - `propose_ngram_draft(history, ngram_size, max_draft_tokens)`
  - `count_accepted_prefix(draft_tokens, target_tokens)`
  - online dry-run / replay stats

### 已有 target verify/commit 原语

- `tools/profile_ngram_commit.py::_target_verify_and_commit()`
  - 为 candidate sequence 预留 speculative append blocks；
  - 构造 proxy block table / slot mapping；
  - 在 KV offload 下 stage read blocks 和 dirty write blocks；
  - 调 `llm.model_runner.run_model(..., is_prefill=True)` 一次性验证 `[last_token] + draft_tokens`；
  - 对 logits 做 argmax，计算 accepted prefix；
  - 调 `BlockManager.commit_accepted_tokens()` 提交 accepted tokens 并释放多余 reserved blocks。

这是 DFlash spike 最应该复用的地方。

### 已有 profiler loop

- `tools/profile_ngram_commit.py` 的 paired / candidate-only loop 已经具备：
  - baseline/candidate 输出对齐；
  - per-prompt stats；
  - commit event 计数；
  - timing breakdown；
  - KV offload stats；
  - JSON summary gate。

DFlash spike 可以先接入这个 profiler，而不是直接改 `LLMEngine.step()`。

## 缺口

### 1. Target hidden state 暴露

DFlash-style draft model 通常需要 target model 的隐藏状态作为条件。当前：

- `ModelRunner.run_model()` 返回 logits；
- `Qwen3Model.forward()` 返回 hidden states；
- `Qwen3ForCausalLM.compute_logits()` 单独把 hidden states 转 logits；
- 但 profiler / engine 层没有稳定 API 暴露 target hidden states。

最小 spike 可加一个只供 profiler 使用的 helper，而不是改公共 decode API：

```python
hidden_states = llm.model_runner.run_model_hidden(input_ids, positions, is_prefill=False)
logits = llm.model_runner.model.compute_logits(hidden_states)
```

注意事项：

- hidden state shape 要对齐 `logits_indices`；
- KV offload 下仍要经过 `_kv_offload_before_forward()` 和 dirty/writeback 流程；
- TP 场景下 hidden/logits 分布式语义需要单独确认，spike 可以先限制 `world_size=1`。

### 2. Draft model 接口

先不要绑定具体 DFlash checkpoint，定义最小接口：

```python
class BlockDraft:
    tokens: list[int]
    scores: list[float] | None
    metadata: dict


class BlockDraftModel:
    def propose_block(
        self,
        history_tokens: list[int],
        target_hidden,
        max_draft_tokens: int,
    ) -> BlockDraft:
        ...
```

第一阶段可以用 toy draft model 模拟 DFlash：

- `EchoBlockDraftModel`：复用 n-gram 或固定 token block；
- `RandomBlockDraftModel`：只验证 plumbing，不追求 acceptance；
- 后续才接真实 diffusion draft checkpoint。

### 3. Verify/commit API 泛化

当前 `_target_verify_and_commit()` 已经够接 DFlash，但命名和输入偏 n-gram。建议抽象为：

```python
def verify_and_commit_block(llm, seq, draft_tokens: list[int], *, source: str, simulate_kv_upload_mb: float = 0.0) -> dict:
    ...
```

并保留事件字段：

- `draft_source`: `ngram` / `dflash_toy` / `dflash`
- `drafted_tokens`
- `accepted_count`
- `target_tokens`
- `timing_ms`
- `reserved_blocks`
- `dirty_blocks`

### 4. Sampling / distribution correctness

当前 profiler 用 greedy argmax 验证，适合 correctness smoke，不等于完整 speculative sampling。

DFlash 完整实现前必须明确：

- 是否只支持 `temperature=0.0` 的 greedy correctness path；
- 若支持 sampling，需要 draft probability / target probability 的接受规则；
- mismatch 后 fallback token 如何采样；
- EOS 和 max_tokens budget 如何处理。

建议 spike 只支持 greedy，保持与当前 n-gram commit smoke 一致。

### 5. 与 KV offload 的交互

DFlash target verify 一次 forward 的 query 长度是 `1 + len(draft_tokens)`，会写入多个 speculative KV slots。当前 `_target_verify_and_commit()` 已经处理：

- reserve append blocks；
- proxy block table；
- dirty block tracking；
- immediate writeback；
- accepted prefix commit；
- unused reserved block release。

需要重点验证的新增边界：

- draft block 跨多个 KV blocks；
- accepted prefix 为 0；
- accepted prefix 落在中间 block；
- EOS 在 accepted prefix 中；
- KV offload staging slots 小于 visible logical blocks 时，是否仍走 blockwise decode/preverify path。

## 建议实施顺序

### Phase 0：文档与接口草案

已完成本文。

### Phase 1：抽象 verify/commit，不改行为（已完成）

目标：

- 把 `_target_verify_and_commit()` 改名/包装成通用 `verify_and_commit_block()`；
- n-gram profiler 继续通过；
- JSON event 增加 `draft_source="ngram"`；
- 不引入 DFlash draft model。

2026-07-07 已落地：

- 新增 draft-source agnostic `verify_and_commit_block()`；
- 保留 `_target_verify_and_commit()` 作为 n-gram 兼容 wrapper；
- target verify event 增加 `draft_source` 字段，当前 n-gram 路径固定为 `"ngram"`；
- 未改 scheduler / runtime 行为。

验证：

- 本地 `tools/test_ngram_speculative.py` 通过；
- 本地 `tools/test_chunked_prefill.py` 通过；
- 远程 Qwen3-0.6B 短 candidate-only smoke 通过：

```bash
CUDA_VISIBLE_DEVICES=7 TINYVLLM_DIST_PORT=34568 MASTER_PORT=34568 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --mode candidate-only \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B \
  --prompt "alpha beta gamma alpha beta gamma alpha beta gamma alpha beta gamma" \
  --max-output-len 4 \
  --temperature 0.0 \
  --ngram-size 3 \
  --max-draft-tokens 2 \
  --max-commit-events 1 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 1 \
  --out-json profile_out/dflash_phase1_ngram_candidate_smoke_20260707.json
```

结果：`gate_pass=true`，`commit_events=1`，`accepted_count=2`，`acceptance_rate=1.0`，`commit_event.draft_source="ngram"`。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD python3 tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD python3 tools/test_chunked_prefill.py
```

### Phase 2：Toy block draft profiler

目标：

- 新增 `--draft-source {ngram,dflash-toy}`；
- `dflash-toy` 复用 toy draft model 产生 block；
- 复用同一个 `verify_and_commit_block()`；
- 只支持 `temperature=0.0`。

2026-07-07 已开始落地：

- `tools/profile_ngram_commit.py` 新增 `--draft-source {ngram,dflash-toy}`；
- `ngram` 继续走原 n-gram helper；
- `dflash-toy` 使用 deterministic `repeat_recent_tokens` toy strategy，只验证 block draft plumbing，不代表真实 DFlash diffusion draft quality；
- commit event 的 `draft_source` 会随 draft source 变化；
- verify event 增加 `draft_metadata`，用于记录 toy strategy 或 n-gram match 信息；
- 新增 `verify_events` 记录所有 target verify attempts，包括 `accepted_count=0` 的 zero-accept plumbing 事件；
- 新增 `--allow-zero-accept`，只供 toy/plumbing smoke 使用，允许没有 accepted tokens 时 gate 通过。

远程 plumbing smoke 已通过：

```bash
CUDA_VISIBLE_DEVICES=7 TINYVLLM_DIST_PORT=34569 MASTER_PORT=34569 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --mode candidate-only \
  --draft-source dflash-toy \
  --allow-zero-accept \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B \
  --prompt "alpha beta gamma alpha beta gamma alpha beta gamma alpha beta gamma" \
  --max-output-len 4 \
  --temperature 0.0 \
  --max-draft-tokens 2 \
  --max-commit-events 1 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 1 \
  --out-json profile_out/dflash_phase2_toy_candidate_smoke_20260707_allow_zero.json
```

结果：`gate_pass=true`，`commit_attempts=3`，`zero_accept_events=3`，`accepted_count=0`，`verify_events` 中包含 `draft_source="dflash-toy"` 和 `draft_metadata.toy_strategy="repeat_recent_tokens"`。这只证明 toy block draft plumbing 和 target verify path 可用，不代表真实 DFlash 接受率。

验证：

- 本地纯 Python helper tests；
- 远程 Qwen3-0.6B candidate-only smoke；
- KV offload off/on 对比。

### Phase 3：Target hidden state extraction

目标：

- 增加 profiler-only hidden state extraction helper；
- 只在 `world_size=1`、greedy、Qwen3 path 下验证；
- 输出 hidden shape / dtype / device 到 JSON debug 字段。

风险：

- 可能触发额外 KV write；
- logits_indices 与 hidden row 对齐容易出错；
- TP 下不一定成立。

### Phase 4：真实 DFlash draft model

只有 Phase 1-3 都稳定后再考虑。

需要新增：

- draft model 权重加载；
- tokenizer / vocab 对齐检查；
- block diffusion sampling schedule；
- draft latency timing；
- target verify acceptance metrics；
- 与 n-gram baseline 的 throughput 对比。

## 不建议立即做的事

- 不要直接把 DFlash 接进 `LLMEngine.step()`；
- 不要在 KV offload correctness 还在优化时引入真实 diffusion checkpoint；
- 不要同时做 DFlash 和 Triton/FlashAttention window kernel；
- 不要在非 greedy sampling 上先行实现 acceptance rule，容易把 correctness 问题扩大。

## 下一步建议

下一步继续 Phase 2：

1. 设计更容易 accepted 的 toy strategy，或进入 target hidden state extraction；
2. 若继续 toy strategy，建议新增 `dflash-toy-ngram-or-repeat`，优先 n-gram 可接受 block，fallback repeat recent tokens；
3. 若进入 hidden extraction，保持 `world_size=1`、greedy、profiler-only，不改 `LLMEngine.step()`。

Phase 2 完成后，再决定是否进入 target hidden state extraction。
