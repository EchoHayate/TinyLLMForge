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

- 新增 `--draft-source {ngram,dflash-toy,dflash-toy-ngram-or-repeat}`；
- `dflash-toy` 复用 toy draft model 产生 block；
- 复用同一个 `verify_and_commit_block()`；
- 只支持 `temperature=0.0`。

2026-07-07 已开始落地：

- `tools/profile_ngram_commit.py` 新增 `--draft-source {ngram,dflash-toy}`；
- `ngram` 继续走原 n-gram helper；
- `dflash-toy` 使用 deterministic `repeat_recent_tokens` toy strategy，只验证 block draft plumbing，不代表真实 DFlash diffusion draft quality；
- `dflash-toy-ngram-or-repeat` 优先复用 n-gram 可接受 block，fallback 到 `repeat_recent_tokens`，用于获得 accepted-friendly toy smoke；
- commit event 的 `draft_source` 会随 draft source 变化；
- verify event 增加 `draft_metadata`，用于记录 toy strategy 或 n-gram match 信息；
- 新增 `verify_events` 记录所有 target verify attempts，包括 `accepted_count=0` 的 zero-accept plumbing 事件；
- 新增 `--allow-zero-accept`，只供 toy/plumbing smoke 使用，允许没有 accepted tokens 时 gate 通过。
- 新增 `--debug-target-hidden`，只在 profiler verify path 中调用 `run_model(..., return_hidden=True)`，把 target hidden `shape/dtype/device` 写到 verify event 的 `target_hidden_debug`。

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

远程 accepted-friendly toy smoke 已通过：

```bash
CUDA_VISIBLE_DEVICES=7 TINYVLLM_DIST_PORT=34570 MASTER_PORT=34570 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --mode candidate-only \
  --draft-source dflash-toy-ngram-or-repeat \
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
  --out-json profile_out/dflash_phase2_toy_hybrid_candidate_smoke_20260707.json
```

结果：`gate_pass=true`，`commit_events=1`，`accepted_count=2`，`acceptance_rate=1.0`，`draft_source="dflash-toy-ngram-or-repeat"`，`draft_metadata.toy_strategy="ngram_or_repeat"`，`draft_metadata.selected_strategy="ngram"`，`match_start=7`，`ngram_size=3`。这说明 accepted-friendly toy 路径能复用同一个 target verify/commit hook，并保留 toy draft source 与 metadata；它仍不是完整 DFlash diffusion draft model。

验证：

- 本地纯 Python helper tests；
- 远程 Qwen3-0.6B candidate-only smoke；
- KV offload off/on 对比。

### Phase 3：Target hidden state extraction

目标：

- 增加 profiler-only hidden state extraction helper；
- 只在 `world_size=1`、greedy、Qwen3 path 下验证；
- 输出 hidden shape / dtype / device 到 JSON debug 字段。

2026-07-07 profiler-only hidden debug 已落地并远程验证：

```bash
CUDA_VISIBLE_DEVICES=7 TINYVLLM_DIST_PORT=34571 MASTER_PORT=34571 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --mode candidate-only \
  --draft-source dflash-toy-ngram-or-repeat \
  --debug-target-hidden \
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
  --out-json profile_out/dflash_phase2_hidden_debug_smoke_20260707.json
```

结果：`gate_pass=true`，`commit_events=1`，`accepted_count=2`，`acceptance_rate=1.0`，`target_hidden_debug.shape=[3, 1024]`，`target_hidden_debug.dtype="torch.bfloat16"`，`target_hidden_debug.device="cuda:0"`。这只在 profiler verify path 中调用 `run_model(..., return_hidden=True)`，不修改 `LLMEngine.step()` 或核心 runtime。

2026-07-07 hidden-to-draft adapter stub 已落地并远程验证：

```bash
CUDA_VISIBLE_DEVICES=7 TINYVLLM_DIST_PORT=34572 MASTER_PORT=34572 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --mode candidate-only \
  --draft-source dflash-toy-ngram-or-repeat \
  --debug-hidden-to-draft-stub \
  --debug-hidden-to-draft-top-k 2 \
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
  --out-json profile_out/dflash_phase3_hidden_to_draft_stub_smoke_20260707.json
```

结果：`gate_pass=true`，`commit_events=1`，`accepted_count=2`，`acceptance_rate=1.0`，`hidden_to_draft_stub.adapter="target_hidden_topk_stub"`，`hidden_to_draft_stub.shape=[3, 1024]`，`hidden_to_draft_stub.dtype="torch.bfloat16"`，`hidden_to_draft_stub.device="cuda:0"`，`hidden_to_draft_stub.top_k=2`，`hidden_to_draft_stub.rows=2`。stub 只记录 target hidden metadata 和 verify logits top-k preview，例如 row 0 top tokens `[13440, 8287]`、row 1 top tokens `[21619, 13440]`；它不采样、不替换 draft tokens、不改变 acceptance rule，也不修改 runtime。

2026-07-07 hidden-to-draft adapter interface schema/timing 已落地并远程验证：

```bash
CUDA_VISIBLE_DEVICES=7 TINYVLLM_DIST_PORT=34573 MASTER_PORT=34573 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --mode candidate-only \
  --draft-source dflash-toy-ngram-or-repeat \
  --debug-hidden-to-draft-stub \
  --debug-hidden-to-draft-top-k 2 \
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
  --out-json profile_out/dflash_phase3_adapter_interface_smoke_20260707.json
```

结果：`gate_pass=true`，`accepted_count=2`，`hidden_to_draft_stub.interface_version=1`，`runtime_mutation=false`，`input_schema.hidden_states.shape=[3, 1024]`，`input_schema.logits.shape=[2, 151936]`，`output_schema` 定义 `draft_token_ids` / `draft_scores` / `num_rows` / `source`，`output.draft_token_ids=[13440, 21619]`，`output.num_rows=2`。新增 timing 字段：`adapter_total_ms=142.16194674372673`，`logits_to_cpu_ms=8.412063121795654`，`topk_ms=133.64192843437195`。注意当前 timing 是 Python profiler preview 成本，尤其 top-k 走 Python sort，不代表未来优化后的 adapter latency；它的价值是先固定 profiler-only ABI 和计时字段。

2026-07-07 可替换 hidden-to-draft adapter 入口与 `linear-stub` skeleton 已落地：

- 新增 `--hidden-to-draft-adapter {topk-stub,linear-stub}`，默认 `topk-stub`；
- `linear-stub` 只改变 profiler JSON 中的 adapter/source/projection/timing 字段，不参与 draft proposal、不替换 accepted tokens、不改 runtime；
- `linear-stub` 当前仍复用 logits top-1 作为 deterministic placeholder output，并额外记录 `linear_projection_ms`，为后续真实 hidden linear projection 预留 ABI。

远程第一次尝试使用 `TINYVLLM_DIST_PORT=34574 MASTER_PORT=34574` 失败，报 `RuntimeError: ... EADDRINUSE, address already in use`。这仍是已知端口占用问题，不是 adapter 代码失败；重跑时换到 `34674` 通过：

```bash
CUDA_VISIBLE_DEVICES=7 TINYVLLM_DIST_PORT=34674 MASTER_PORT=34674 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --mode candidate-only \
  --draft-source dflash-toy-ngram-or-repeat \
  --debug-hidden-to-draft-stub \
  --hidden-to-draft-adapter linear-stub \
  --debug-hidden-to-draft-top-k 2 \
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
  --out-json profile_out/dflash_phase3_linear_stub_interface_smoke_20260707.json
```

结果：`gate_pass=true`，`accepted_count=2`，`hidden_to_draft_adapter="linear-stub"`，`hidden_to_draft_stub.adapter="target_hidden_linear_stub"`，`input_schema.adapter="linear-stub"`，`output_schema.projection="deterministic_placeholder"`，`output.source="target_hidden_linear_stub"`，`output.draft_token_ids=[13440, 21619]`。新增/确认 timing：`adapter_total_ms=126.10501796007156`，`logits_to_cpu_ms=8.051183074712753`，`linear_projection_ms=0.0002868473529815674`，`topk_ms=117.97097697854042`。当前 `linear_projection_ms` 是 no-op placeholder 边界计时，下一步可替换成真实 deterministic linear projection skeleton。

2026-07-07 `linear-stub` 已从 no-op placeholder 替换为 deterministic hidden projection skeleton：

- 仍然只在 profiler-only `hidden_to_draft_stub` 中记录，不参与 draft proposal、不替换 accepted tokens、不改变 acceptance/commit 行为，也不接入 `LLMEngine.step()`；
- 先把 target hidden rows 搬到 CPU，再用固定 seed 公式对一个小 vocab candidate set 做 deterministic pseudo linear projection；
- candidate set 当前取前 8 个 token id，即 `[0,1,2,3,4,5,6,7]`；伪权重公式为 `(((dim_index + 1) * (candidate_index + 3) + 17) % 11 - 5) / 4.0`；
- 输出 schema 从 `deterministic_placeholder` 改为 `deterministic_hidden_linear_stub`，并新增 `projection_metadata`、`hidden_to_cpu_ms`、`linear_projection_ms`；
- 修复了 hidden rows 与 logits rows 不一致时 `rows` / `output.num_rows` 仍按 logits rows 计数的问题；现在 `linear-stub` 以 hidden projection preview 行数为准。

远程第一次用 `TINYVLLM_DIST_PORT=34676 MASTER_PORT=34676` 运行 Qwen3 smoke 失败，报：

```text
RuntimeError: The server socket has failed to listen on any local network address. useIpv6: 0, code: -98, name: EADDRINUSE, message: address already in use
```

这仍是远程端口占用，不是 deterministic projection 代码失败；换到 `35676` 后通过：

```bash
CUDA_VISIBLE_DEVICES=7 TINYVLLM_DIST_PORT=35676 MASTER_PORT=35676 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --mode candidate-only \
  --draft-source dflash-toy-ngram-or-repeat \
  --debug-hidden-to-draft-stub \
  --hidden-to-draft-adapter linear-stub \
  --debug-hidden-to-draft-top-k 2 \
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
  --out-json profile_out/dflash_phase3_linear_projection_stub_smoke_20260707_r2.json
```

结果：`gate_pass=true`，`accepted_count=2`，`hidden_to_draft_stub.adapter="target_hidden_linear_stub"`，`output_schema.projection="deterministic_hidden_linear_stub"`，`projection_metadata.seed=17`，`projection_metadata.candidate_token_ids=[0,1,2,3,4,5,6,7]`，`projection_metadata.hidden_dim=1024`，`projection_metadata.candidate_count=8`，`output.draft_token_ids=[0,0,7]`，`output.num_rows=3`，`rows=3`，`input_schema.hidden_states.shape=[3,1024]`，`input_schema.logits.shape=[3,151936]`。timing：`adapter_total_ms=12.286126613616943`，`logits_to_cpu_ms=8.55349749326706`，`hidden_to_cpu_ms=0.17523393034934998`，`linear_projection_ms=3.5200342535972595`，`topk_ms=3.5200342535972595`。这些数字仍是 Python profiler skeleton 成本，只用于 ABI/数据流验证，不代表未来 optimized adapter latency。

2026-07-07 已补 `topk-stub` vs `linear-stub` 3x3 remote compare，用同一 Qwen3-0.6B、同一 prompt、`--draft-source dflash-toy-ngram-or-repeat`、`--debug-hidden-to-draft-top-k 2`。第一次复用固定端口序列时第二个进程仍遇到 `EADDRINUSE`；随后改为每次远程动态探测空闲端口并把 stdout/stderr 写入同名 `.log`，6 次 JSON 均成功输出：

- `profile_out/dflash_phase3_adapter_compare_topk-stub_r{1,2,3}_20260707_cmp2.json`
- `profile_out/dflash_phase3_adapter_compare_linear-stub_r{1,2,3}_20260707_cmp2.json`

结果概览：

| adapter | runs | gate | output rows | draft ids | projection | adapter_total_ms mean/stdev | key timing |
| --- | ---: | --- | ---: | --- | --- | ---: | --- |
| `topk-stub` | 3 | all `gate_pass=true` | 2 | `[13440,21619]` | `logits_topk` | `124.903983 / 1.449532` | `logits_to_cpu_ms=8.816361±0.275745`, `topk_ms=116.005514±1.719655` |
| `linear-stub` | 3 | all `gate_pass=true` | 3 | `[0,0,7]` | `deterministic_hidden_linear_stub` | `12.578132 / 0.081115` | `logits_to_cpu_ms=8.753903±0.075491`, `hidden_to_cpu_ms=0.191076±0.019228`, `linear_projection_ms=3.593331±0.057019` |

ABI 结论：当前 `hidden_to_draft_stub` 已足够支撑下一步真实 draft model stub 的 profiler-only 接入：`interface_version`、`runtime_mutation=false`、`input_schema.hidden_states`、`input_schema.logits`、`input_schema.adapter`、`output_schema`、`output`、`timing_ms` 都稳定存在；`linear-stub` 额外有 `projection_metadata`，能表达 seed、candidate token set、hidden dim 和 candidate count。注意两条路径的 `rows` 语义目前不同：`topk-stub` 预览 verify logits rows，因此为 2；`linear-stub` 预览 target hidden rows，因此为 3。这不是 correctness failure，但真实 draft model stub 前最好把字段命名进一步显式化，例如增加 `input_schema.hidden_rows` / `input_schema.logit_rows` / `output.projected_rows`，避免后续读 JSON 时误解。

Timing 结论：字段足够继续做趋势对比，但仍不能代表真实 adapter latency。`topk-stub` 的 `topk_ms` 是对完整 vocab logits 做 Python sort，主要用于 baseline preview，成本约 116ms；`linear-stub` 的 `linear_projection_ms` 只投影 3x1024 到 8 个 candidate，成本约 3.59ms，且当前实现把 `topk_ms` 复用为 projection 排序耗时。下一步如果接真实 draft model stub，建议把 timing 拆成 `hidden_to_cpu_ms`、`draft_model_forward_ms`、`candidate_select_ms`、`adapter_total_ms`，并保留 `runtime_mutation=false` 作为 profiler-only 安全闸。

2026-07-07 已继续显式化 adapter ABI/timing 字段：

- `input_schema` 新增 `hidden_rows`、`logit_rows`、`projected_rows`，并修正 `input_schema.logits.shape` 始终表达真实 logits preview shape，不再被 `linear-stub` 的 hidden projected rows 覆盖；
- `output_schema` / `output` 新增 `projected_rows`，保留旧的 `rows` / `num_rows` 兼容现有分析脚本；
- `timing_ms` 新增 `draft_model_forward_ms` 与 `candidate_select_ms`，当前 profiler stub 中 `draft_model_forward_ms=0.0`；`candidate_select_ms` 先等价于当前候选选择/排序阶段耗时，后续真实 draft model stub 可把模型 forward 与候选选择分开记录；
- 本地测试覆盖了 `linear-stub` 在 hidden rows 与 logits rows 不一致时的 schema：`hidden_rows=3`、`logit_rows=2`、`projected_rows=3`、`input_schema.logits.shape=[2,3]`。

远程同步后已用 `profile_out/dflash_phase3_adapter_abi_fields_smoke_20260707.json` 验证：`gate_pass=true`、`accepted_count=2`、`hidden_rows=3`、`logit_rows=2`、`projected_rows=3`、`input_schema.logits.shape=[2,151936]`、`output.projected_rows=3`，`timing_ms` 包含 `candidate_select_ms` 与 `draft_model_forward_ms`。实测 timing：`adapter_total_ms=12.19320297241211`、`logits_to_cpu_ms=8.414741605520248`、`hidden_to_cpu_ms=0.17217546701431274`、`linear_projection_ms=3.570154309272766`、`candidate_select_ms=3.570154309272766`、`draft_model_forward_ms=0.0`。

2026-07-07 已新增 profiler-only `draft-model-stub` adapter：

- CLI 新增 `--hidden-to-draft-adapter draft-model-stub`；
- `hidden_to_draft_stub.adapter="target_hidden_draft_model_stub"`，`runtime_mutation=false`，`output_schema.projection="deterministic_draft_model_stub"`；
- stub 从 target hidden rows 生成 deterministic candidate logits / candidate token ids，并写入 `output.candidate_token_ids`、`output.candidate_logits`、`draft_model_metadata`；
- `timing_ms.draft_model_forward_ms` 被真实占用为 pseudo draft model forward 计时，`candidate_select_ms` 记录候选排序/选择阶段；
- 仍然不把 stub 输出接入 draft proposal、target verify、acceptance 或 commit；远程 smoke 中 `commit_event.draft_tokens` 与 `accepted_tokens` 仍来自 `dflash-toy-ngram-or-repeat`。

远程验证记录：第一次在 `CUDA_VISIBLE_DEVICES=7` 上失败于模型初始化 `assert auto_num_blocks > 0`，根因是 GPU7 当时显存占用约 69GiB/80GiB，不是 adapter 代码失败。改用空闲 GPU3 后通过：`profile_out/dflash_phase3_draft_model_stub_smoke_20260707_gpu3.json`，`gate_pass=true`、`accepted_count=2`、`adapter="target_hidden_draft_model_stub"`、`projection="deterministic_draft_model_stub"`、`hidden_rows=3`、`logit_rows=2`、`projected_rows=3`、`output.projected_rows=3`、`output.draft_token_ids=[7,7,7]`、第一行 candidate token ids `[7,1]`。metadata：`seed=23`、`candidate_token_ids=[0,1,2,3,4,5,6,7]`、`hidden_dim=1024`、`candidate_count=8`、`stub_version=1`。timing：`adapter_total_ms=12.383360415697098`、`logits_to_cpu_ms=8.444327861070633`、`hidden_to_cpu_ms=0.26154518127441406`、`draft_model_forward_ms=3.5831667482852936`、`candidate_select_ms=0.021237879991531372`、`topk_ms=3.6189667880535126`。

2026-07-07 已补 3-way adapter compare（GPU4，Qwen3-0.6B，同一 prompt，3 runs/adapter）：

- 输出：`profile_out/dflash_phase3_adapter_3way_compare_{topk-stub,linear-stub,draft-model-stub}_r{1,2,3}_20260707_cmp3.json`；
- 9 次均 `gate_pass=true`、`accepted_count=2`，commit 的 `draft_tokens` / `accepted_tokens` 均为 `[13440,21619]`，说明三个 profiler stub 都未改变 acceptance/runtime；
- `draft-model-stub` 的 `candidate_token_ids`、`candidate_logits`、`draft_token_ids`、`draft_model_metadata` 三次完全稳定：`draft_token_ids=[7,7,7]`、每行 candidate ids 均 `[7,1]`，metadata 为 `seed=23`、candidate set `[0,1,2,3,4,5,6,7]`、`hidden_dim=1024`、`candidate_count=8`、`stub_version=1`。

| adapter | stable output | adapter_total_ms mean/stdev | key timing mean/stdev |
| --- | --- | ---: | --- |
| `topk-stub` | `draft_token_ids=[13440,21619]` | `131.355740 / 6.488973` | `candidate_select_ms=122.164890±6.773997`, `logits_to_cpu_ms=9.093589±0.297199` |
| `linear-stub` | `draft_token_ids=[0,0,7]` | `12.964208 / 0.502438` | `candidate_select_ms=3.640130±0.087377`, `hidden_to_cpu_ms=0.357489±0.036887` |
| `draft-model-stub` | `draft_token_ids=[7,7,7]`, candidate ids/logits stable | `13.104054 / 0.257300` | `draft_model_forward_ms=3.721040±0.024405`, `candidate_select_ms=0.021578±0.001956`, `hidden_to_cpu_ms=0.349854±0.030066` |

结论：当前 `draft-model-stub` 的 ABI 和输出稳定性足够继续向“真实 draft model 接口”抽象，但还不应接入 runtime。下一步若继续，应把 deterministic pseudo forward 的输入/输出边界抽成独立函数或类，例如 `run_draft_model_stub(hidden_rows, candidate_token_ids, top_k) -> {candidate_logits, candidate_token_ids, timing}`，再替换为真实 draft model forward 时只改该边界；同时保持 profiler-only `runtime_mutation=false` gate。

2026-07-08 已把 deterministic pseudo forward 抽成 `run_draft_model_stub(hidden_rows, candidate_token_ids, top_k)`：

- 输出边界固定为 `candidate_token_ids`、`candidate_logits`、`draft_token_ids`、`draft_scores`、`preview`、`metadata`、`timing_ms.{draft_model_forward_ms,candidate_select_ms}`；
- `summarize_hidden_to_draft_stub(..., adapter="draft-model-stub")` 复用该 helper 输出，保持原 profiler JSON schema 不变；
- 本地测试直接覆盖 helper 的 deterministic output，并验证 summarize 路径复用同一 candidate ids/logits/metadata；
- 远程 `profile_out/dflash_phase3_draft_model_stub_boundary_smoke_20260708.json` 通过：`gate_pass=true`、`accepted_count=2`、`draft_token_ids=[7,7,7]`、第一行 candidate ids `[7,1]`、`draft_model_forward_ms=3.5160109400749207`、`candidate_select_ms=0.015269964933395386`；commit `draft_tokens`/`accepted_tokens` 仍为 `[13440,21619]`，确认抽象后仍不影响 runtime。

2026-07-08 继续把 helper 边界包成更接近真实 draft model 的 dataclass/config shell：

- 新增 `DraftModelStubConfig(seed=23, stub_version=1)` 和 `DraftModelResult(candidate_token_ids, candidate_logits, draft_token_ids, draft_scores, preview, metadata, timing_ms).to_dict()`；
- `run_draft_model_stub(..., config=None) -> DraftModelResult`，并在边界层显式校验 empty candidate set 与 ragged hidden rows，未来替换真实 draft model forward 时可复用同一错误边界；
- 本地测试覆盖 dataclass `to_dict()`、config metadata propagation、边界错误，以及 `summarize_hidden_to_draft_stub(..., adapter="draft-model-stub")` 继续读取 result 字段；
- 远程 `profile_out/dflash_phase3_draft_model_dataclass_shell_smoke_20260708.json` 通过：`gate_pass=true`、`accepted_count=2`、`runtime_mutation=false`、`input_schema.adapter="draft-model-stub"`、`output.draft_token_ids=[7,7,7]`、第一行 candidate ids `[7,1]`、`draft_model_metadata.stub_version=1`、`draft_model_forward_ms=4.899017512798309`、`candidate_select_ms=0.01652538776397705`；commit `draft_tokens`/`accepted_tokens` 仍为 `[13440,21619]`，确认 dataclass shell 仍只是 profiler-only ABI，不参与 proposal/acceptance/runtime。

2026-07-08 又补齐输入侧 contract：

- 新增 `DraftModelInput.from_rows(...).to_dict()/schema()`，把 `hidden_rows`、`candidate_token_ids`、`top_k`、`source_shape`、`source_dtype`、`source_device` 显式作为 draft model forward 输入边界；
- `run_draft_model_stub()` 保持旧签名兼容，同时可直接接收 `DraftModelInput`；`draft_model_metadata.input_schema` 记录输入边界，便于未来真实 draft model forward 检查 hidden 来源和 top-k/candidate contract；
- 远程 `profile_out/dflash_phase3_draft_model_input_contract_smoke_20260708.json` 通过：`gate_pass=true`、`accepted_count=2`、`runtime_mutation=false`、`draft_model_metadata.input_schema={hidden_rows=3, hidden_dim=1024, candidate_count=8, top_k=2, source_shape=[3,1024], source_dtype="torch.bfloat16", source_device="cuda:0"}`、`output.draft_token_ids=[7,7,7]`、第一行 candidate ids `[7,1]`、`draft_model_forward_ms=5.126491189002991`、`candidate_select_ms=0.02197548747062683`；commit `draft_tokens`/`accepted_tokens` 仍为 `[13440,21619]`。

2026-07-08 多 prompt / batch shape smoke 已验证：

- 远程 `profile_out/dflash_phase3_draft_model_batch_shape_smoke_20260708.json` 使用两个 prompt、`--max-num-seqs 2`、`draft-model-stub`、`top_k=2`；
- `gate_pass=true`、`num_prompts=2`、`commit_events=2`、`accepted_count=4`，每个 prompt 各 `commit_events=1`、`verify_events=1`、`accepted_count=2`；
- 两个 event 的 `draft_model_metadata.input_schema` 都独立记录 `hidden_rows=3`、`hidden_dim=1024`、`candidate_count=8`、`top_k=2`、`source_shape=[3,1024]`、`source_dtype="torch.bfloat16"`、`source_device="cuda:0"`；
- prompt 0 的 commit `draft_tokens/accepted_tokens=[13440,21619]`、`output.draft_token_ids=[7,7,7]`、第一行 candidate ids `[7,1]`；prompt 1 的 commit `draft_tokens/accepted_tokens=[6303,6176]`、`output.draft_token_ids=[1,2,2]`、第一行 candidate ids `[1,2]`。这确认 batch 场景下 event 级 DraftModelInput schema 没有 prompt 之间串写，且仍保持 `runtime_mutation=false`。

2026-07-08 已把 profiler-only draft model schema 抽成小模块：

- 新增 `tools/draft_model_schema.py`，承载 `DraftModelInput`、`DraftModelResult`、`DraftModelStubConfig`；`tools/profile_ngram_commit.py` 只导入这些 schema/dataclass，避免继续在大 profiler 文件里堆接口定义；
- 本地 `tools/test_ngram_speculative.py` 直接加载 `draft_model_schema`，验证模块 API 与 profiler 使用的是同一组 dataclass；
- 远程 `profile_out/dflash_phase3_draft_model_schema_module_smoke_20260708.json` 通过：`gate_pass=true`、`accepted_count=2`、`runtime_mutation=false`、`DraftModelInput` schema 仍为 `source_shape=[3,1024]`、`source_dtype="torch.bfloat16"`、`source_device="cuda:0"`，commit `draft_tokens/accepted_tokens=[13440,21619]` 不变。

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

当前 Phase 2 / Phase 3 profiler-only 路径已完成最小可验证闭环：

1. `--draft-source dflash-toy-ngram-or-repeat` 远程通过，确认有 accepted tokens 且 `draft_source` 字段保留 toy source；
2. `--debug-target-hidden` 远程通过，确认 `target_hidden_debug` 记录 shape/dtype/device；
3. `--debug-hidden-to-draft-stub` 远程通过，确认 profiler 能把 target hidden metadata、输入/输出 schema、adapter timing 和 verify logits top-k preview 写入 `hidden_to_draft_stub`；
4. `--hidden-to-draft-adapter linear-stub` 远程通过，确认 adapter selector、linear-stub output schema 和 `linear_projection_ms` 已进入 profiler JSON；
5. deterministic hidden projection skeleton 远程通过，确认 `linear-stub` 可以从 hidden rows 投影到小 vocab candidate set，并记录 projection metadata / hidden copy timing / projection timing，且仍不参与 acceptance/runtime；
6. `topk-stub` vs `linear-stub` 3x remote compare 已通过，确认 ABI 字段稳定、timing 字段足够继续 profiler-only 真实 draft model stub；后续应先显式化 row-count 字段和拆分 draft-model timing，再考虑完整 diffusion checkpoint。
7. `draft-model-stub` 已完成 profiler-only dataclass/config shell：真实 draft model 未来需要返回的 candidate ids/logits、draft tokens/scores、metadata、timing 都已经在 `DraftModelResult` 中显式化，并有本地错误边界测试与远程 Qwen3 smoke 验证。
8. `DraftModelInput` 已补齐输入侧 contract，并通过多 prompt / batch shape smoke 证明 event 级 schema 不混淆。
9. profiler-only draft model schema 已抽成 `tools/draft_model_schema.py` 小模块，下一步若继续 Phase 3，最有价值的是补更多形状覆盖或做真实 draft model 接入前置检查（vocab/tokenizer/hidden_dim contract），仍不应接入真实 checkpoint 或 runtime。

仍不建议直接接入 `LLMEngine.step()`；真实 DFlash draft model 接入前，应继续保持 profiler-only、greedy、`world_size=1` 范围。
