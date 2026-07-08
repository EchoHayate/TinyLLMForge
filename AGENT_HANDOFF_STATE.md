# Agent Handoff State

> 目的：上下文中断后，新的 agent 先读这个文件，避免重新猜工作区、远程环境和当前任务状态。

## 必须优先使用远程环境

- 远程机器：`sitian@10.232.195.203`
- 远程项目目录：`/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge`
- 远程 Python：`/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`
- Python 版本：`Python 3.11.15`
- Torch：`2.4.1+cu121`
- CUDA：远程 Python 下 `torch.cuda.is_available() == True`
- 默认 GPU：`CUDA_VISIBLE_DEVICES=7`
- Conda/Miniforge：`/data00/home/sitian/sitian-workspace01/tllm/miniforge/bin/conda`
- Conda envs 中包含：
  - `/data00/home/sitian/sitian-workspace01/tllm/env`
  - `/data00/home/sitian/sitian-workspace01/tllm/miniforge`（base）
  - `/data00/home/sitian/miniconda3`
  - `/data00/home/sitian/miniconda3/envs/py311`

本地桌面项目入口是 `/Users/bytedance/Desktop/TinyLLMForge`（用户口径：Tiny LLM Forge）。这个路径是后续 agent 应优先读取/进入的本地项目路径；涉及 GPU、torch CUDA、profile、needle、Qwen 模型时仍必须通过 SSH 在远程跑，不要用本地 `/opt/homebrew/bin/python3.12` 代替。

## 本地路径

- 本地桌面项目入口：`/Users/bytedance/Desktop/TinyLLMForge`（用户口径：Tiny LLM Forge）
- 实际 symlink 指向：`/Users/bytedance/dev/TinyLLMForge`
- 后续 agent 本地读文件时优先用桌面路径：`/Users/bytedance/Desktop/TinyLLMForge/AGENT_HANDOFF_STATE.md`
- 本地分支：`feat/kv-sparse-attention`（本地 git 仓库）
- 远程项目目录当前看起来不是 git worktree（`git status` 报 not a git repository），更像同步后的运行目录。

## 常用远程命令模板

```bash
ssh sitian@10.232.195.203 'cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge && CUDA_VISIBLE_DEVICES=7 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python <script> <args>'
```

验证远程 Python/CUDA：

```bash
ssh sitian@10.232.195.203 'cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge && /data00/home/sitian/sitian-workspace01/tllm/env/bin/python - <<"PY"
import torch, sys
print(sys.executable)
print(torch.__version__, torch.cuda.is_available())
PY'
```

## 模型和重要产物路径

- Qwen3-8B：`/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B`
- Qwen3-0.6B：`/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B`
- SQ scale 常用：
  - `/tmp/sq_scales_qwen3_8b_a0.85.pt`
  - `/tmp/sq_scales_qwen3_8b_layer_adaptive_floor085.pt`
- 远程 profile 输出目录：`/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/profile_out`
- 远程 needle 输出目录：`/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/needle_sq_results`

## 当前任务主线

最近主线是 TinyLLMForge 的 Qwen3/long-context runtime 实验，尤其：

1. n-gram speculative decoding 原型：S1 online dry-run、S2 target verification、S3 accepted-token KV commit、S4 online benchmark。
2. mixed prefill admission policy：短 prompt 不进入 mixed prefill+decode batch，保护 decode latency。
3. hidden-state / latent projector 相关实验已经有大量脚本和结果，但当前更像封档/实验记录，不是 runtime 主线。

本地最近相关文件：

- `tinyvllm/speculative/ngram.py`
- `tinyvllm/engine/block_manager.py`
- `tinyvllm/engine/llm_engine.py`
- `tinyvllm/engine/scheduler.py`
- `tools/profile_ngram_online.py`
- `tools/profile_ngram_verify.py`
- `tools/profile_ngram_commit.py`
- `tools/test_ngram_speculative.py`
- `tools/test_chunked_prefill.py`
- `tools/profile_chunked_prefill.py`
- `tools/train_latent_projector.py`
- `docs/qwen3-8b-fixes.md`

## 当前已做的本地修复

在本地 `/Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py` 已修两个边界：

1. `_target_verify_and_commit()` 中 `commit_accepted_tokens()` 接管/释放 reserved blocks 后，把本地 `reserved_blocks` 清空，避免异常路径 double release。
2. `run_paired_profile()` 和 `run_candidate_only_profile()` 中，commit 后如果 `llm.is_finished()`，直接 break，避免所有请求已结束后继续 `llm.step()` 触发 scheduler 空队列 assert。

本地已通过：

```bash
git -C /Users/bytedance/dev/TinyLLMForge diff --check
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge python3 /Users/bytedance/dev/TinyLLMForge/tools/test_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge python3 /Users/bytedance/dev/TinyLLMForge/tools/test_ngram_speculative.py
```

注意：这些是本地 CPU/语法检查。GPU/profile/模型相关验证必须在远程跑。

## 建议下一步

1. 后续 agent 先读本文件：`/Users/bytedance/Desktop/TinyLLMForge/AGENT_HANDOFF_STATE.md`。
2. 如果要继续 S4 n-gram speculative，先确认本地改动是否已同步到远程运行目录。
3. 在远程用 Qwen3-0.6B 先跑最小 candidate-only / baseline-only smoke，再跑 8B。
4. 不要把 `.agents/`、`.codex/`、`.pt` checkpoint 这类本地配置/产物默认提交。
5. 然后读取 `docs/qwen3-8b-fixes.md` 最新章节。

## 2026-06-29 远程 S4 smoke 结果

已把本地 `tools/profile_ngram_commit.py` 的边界修复同步到远程运行目录，并在远程通过：

```bash
cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
PYTHONDONTWRITEBYTECODE=1 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py --help >/dev/null
```

Qwen3-0.6B / `max_output_len=32` / `n5d4` smoke：

- candidate-only JSON：`profile_out/ngram_spec_s4_06b_candidate_n5d4_smoke_20260629.json`
- candidate-only log：`profile_out/ngram_spec_s4_06b_candidate_n5d4_smoke_20260629.log`
- baseline-only JSON：`profile_out/ngram_spec_s4_06b_baseline_n5d4_smoke_20260629.json`
- baseline-only log：`profile_out/ngram_spec_s4_06b_baseline_n5d4_smoke_20260629.log`

结果摘要：

| mode | gate_pass | output_tokens | elapsed_s | tok/s | decode_steps | commit_events | accepted_count | drafted_tokens | acceptance_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate-only | true | 32 | 55.705 | 0.574 | 22 | 3 | 9 | 16 | 0.5625 |
| baseline-only | true | 32 | 57.078 | 0.561 | 31 | - | - | - | - |

纯 decode step 汇总：candidate 22 step / 22.875s，baseline 31 step / 24.044s。candidate 确实少跑 9 个 decode step，但单 step 平均更重；这个 smoke 只说明修复后正确性 gate 通过，性能结论还需要更长输出、多 prompt、热身后再判断。

## 2026-06-29 warmup 后的 S4 smoke 结果

结论：之前 55s 主要不是 decode 慢，而是冷启动/首次 shape 编译计入了测量。典型冷启动：prefill 约 32-33s，首个 decode 约 22s；warmup 后测量段恢复到约 1.1s。

已给 `tools/profile_ngram_commit.py` 增加参数：

```bash
--warmup-output-len N
```

它会在正式计时前先用同一个 LLM 跑一个 untimed warmup request，避免把 CUDA / kernel / symbolic shape setup 计入 profile 时间。远程已同步并通过 `py_compile` / `--help`。

Qwen3-0.6B / `max_output_len=32` / `warmup_output_len=4` / `n5d4`：

| mode | gate_pass | warmup_s | measured_elapsed_s | tok/s | decode_steps | commit_events | accepted_count | drafted_tokens | acceptance_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-only | true | 55.045 | 1.171 | 27.33 | 31 | - | - | - | - |
| candidate-only | true | 56.438 | 1.117 | 28.65 | 22 | 3 | 9 | 16 | 0.5625 |

纯 decode 段：baseline 31 step / 1127ms，candidate 22 step / 833ms。candidate 少 9 个 decode step，warmup 后测得约 1.05x wall-clock speedup；样本太短，只能作为正确性/趋势 smoke。

注意：不要在同一默认 `TINYVLLM_DIST_PORT=2333` 下并发跑多个 TinyLLM 进程，否则会报 `EADDRINUSE`。如必须并发，显式设置不同 `TINYVLLM_DIST_PORT`，但同一 GPU 并发会污染性能结果。

## 2026-06-29 target verify 分段计时

已给 `tools/profile_ngram_commit.py` 的 `_target_verify_and_commit()` 增加分段计时，并写入每个 commit event 的 `timing_ms`，summary 里聚合为 `verify_timing_ms`。字段包括：

- `reserve_blocks_ms`
- `verify_prepare_ms`
- `target_forward_ms`
- `accept_sample_ms`
- `commit_metadata_ms`
- `finish_check_ms`
- `verify_commit_total_ms`

远程已同步并通过 `py_compile` / `--help`。计时 smoke 输出：

- JSON：`profile_out/ngram_spec_s4_06b_candidate_n5d4_timing_smoke_20260629.json`
- log：`profile_out/ngram_spec_s4_06b_candidate_n5d4_timing_smoke_20260629.log`

Qwen3-0.6B / `max_output_len=32` / `warmup_output_len=4` / `n5d4` / candidate-only：

| metric | value |
|---|---:|
| gate_pass | true |
| elapsed_s | 1.062 |
| output_tokens_per_s | 30.13 |
| decode_steps | 22 |
| commit_attempts | 4 |
| commit_events | 3 |
| accepted_count | 9 |
| drafted_tokens | 16 |
| acceptance_rate | 0.5625 |

聚合 verify timing（包含 4 次 attempt，其中 1 次 zero-accept）：

| timing | ms |
|---|---:|
| reserve_blocks_ms | 0.025 |
| verify_prepare_ms | 1.848 |
| target_forward_ms | 242.415 |
| accept_sample_ms | 1.831 |
| commit_metadata_ms | 0.064 |
| finish_check_ms | 0.031 |
| verify_commit_total_ms | 246.250 |

接受事件每次约 40ms target forward；metadata commit 基本可忽略。当前主要额外成本在 target forward verify，而不是 block reservation / commit metadata。

## 2026-06-29 长上下文下验证 KV 读取摊薄

按“重点看 target verify forward 是否能在长 context/cache pressure 下摊薄 KV 读取”的口径，跑了 3593-token prompt（`alpha beta gamma ...` 重复 512 次），`max_model_len=8192`，`max_output_len=64`，`warmup_output_len=4`，Qwen3-0.6B，`n5d4`。

输出文件：

- baseline JSON：`profile_out/ngram_spec_s4_06b_baseline_n5d4_long3593_20260629.json`
- baseline log：`profile_out/ngram_spec_s4_06b_baseline_n5d4_long3593_20260629.log`
- candidate JSON：`profile_out/ngram_spec_s4_06b_candidate_n5d4_long3593_20260629.json`
- candidate log：`profile_out/ngram_spec_s4_06b_candidate_n5d4_long3593_20260629.log`

结果：

| mode | gate_pass | output_tokens | elapsed_s | tok/s | decode_steps | accepted | drafted | acceptance_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-only | true | 64 | 2.531 | 25.29 | 63 | - | - | - |
| candidate-only | true | 64 | 1.266 | 50.54 | 16 | 47 | 52 | 0.9038 |

分解：

| mode | prefill_ms | decode_steps | decode_ms | avg_decode_ms | verify_target_forward_ms | verify_total_ms |
|---|---:|---:|---:|---:|---:|---:|
| baseline-only | 179.5 | 63 | 2350.8 | 37.3 | - | - |
| candidate-only | 69.2 | 16 | 643.9 | 40.2 | 531.7 | 542.1 |

关键结论：

- candidate 把 decode step 从 63 降到 16，接受了 47 个 token。
- target verify forward 总计 531.7ms，验证 52 个 draft token，平均约 10.2ms/draft token；baseline decode 平均约 37.3ms/token。
- 在这种长上下文、重复 prompt、高 acceptance 的场景下，target verify 基本表现为“一次约 40ms verify 4 个 token”，确实摊薄了长 prefix KV 读取/attention 成本。
- 这还不是 CPU→GPU KV offload/upload 压力测试；它验证的是 GPU KV 长上下文读取摊薄。若要验证“内存 upload”收益，需要再构造 KV offload/page migration 场景。

## 2026-06-29 模拟 CPU→GPU KV upload pressure

当前 TinyLLMForge 没有真正的 KV CPU offload/page migration；已有 `cpu_offload` 是 decoder layer weight offload，不是 KV offload。因此先给 `tools/profile_ngram_commit.py` 增加了一个可控模拟参数：

```bash
--simulate-kv-upload-mb 128
```

含义：每次 baseline decode step、candidate normal decode step、candidate target verify 前，额外做一次 pinned CPU → GPU copy，用来模拟每轮需要从 CPU/host memory upload KV pages 的成本。脚本会先做一次 upload warmup，避免首次 pin/copy 初始化污染计时。远程已同步并通过 `py_compile` / `--help`。

测试口径：Qwen3-0.6B，3593-token repeated prompt，`max_model_len=8192`，`max_output_len=64`，`warmup_output_len=4`，`n5d4`，`simulate_kv_upload_mb=128`。

输出：

- baseline JSON：`profile_out/ngram_spec_s4_06b_baseline_n5d4_upload128_long3593_20260629.json`
- baseline log：`profile_out/ngram_spec_s4_06b_baseline_n5d4_upload128_long3593_20260629.log`
- candidate JSON：`profile_out/ngram_spec_s4_06b_candidate_n5d4_upload128_long3593_20260629.json`
- candidate log：`profile_out/ngram_spec_s4_06b_candidate_n5d4_upload128_long3593_20260629.log`

结果：

| mode | gate_pass | output_tokens | elapsed_s | tok/s | decode_steps | commit_events | accepted | drafted | acceptance_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-only | true | 64 | 2.721 | 23.52 | 63 | - | - | - | - |
| candidate-only | true | 64 | 1.334 | 47.97 | 16 | 13 | 47 | 52 | 0.9038 |

upload 模拟成本分解：

| mode | normal decode upload ms | verify upload ms | total simulated upload ms |
|---|---:|---:|---:|
| baseline-only | 334.5 | - | 334.5 |
| candidate-only | 84.6 | 68.9 | 153.5 |

关键结论：

- baseline 需要 63 次 decode upload；candidate 只需要 16 次 normal decode upload + 13 次 verify upload。
- 在每轮模拟 128MiB H2D upload 时，candidate 的 simulated upload 总成本从 334.5ms 降到 153.5ms，约减少 54%。
- 总体 wall-clock：baseline 2.721s，candidate 1.334s，约 2.04x。
- 这支持“draft + target verify 可以摊薄 KV upload/page migration 成本”的方向，但注意这仍是 copy 模拟；真正 KV offload 需要实现按 block/page 的 CPU resident KV + prefetch/upload + GPU staging cache。

后续继续推进时，已给 `tools/profile_ngram_commit.py` 的 JSON `summary` 补充模拟 upload 聚合字段，避免每次手动从 `step_records` 和 `verify_timing_ms` 里加总：

- `simulate_kv_upload_mb`
- `normal_decode_simulated_kv_upload_ms`
- `verify_simulated_kv_upload_ms`
- `total_simulated_kv_upload_ms`
- `normal_decode_simulated_kv_upload_events`
- `verify_simulated_kv_upload_events`
- `total_simulated_kv_upload_events`
- `normal_decode_simulated_kv_upload_mib`
- `verify_simulated_kv_upload_mib`
- `total_simulated_kv_upload_mib`

本地已通过：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge python3 /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py --help >/dev/null
```

远程已同步并通过：

```bash
cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
PYTHONDONTWRITEBYTECODE=1 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py --help >/dev/null
```

## 2026-06-30 KV offload MVP-0

按“先做真实 KV offload 的最小闭环，不直接上完整 page migration”的口径，已增加默认关闭的 MVP-0：

- `Config.kv_offload_mvp0`
- `Config.kv_offload_gpu_blocks`
- `Config.kv_offload_logical_blocks`
- `tools/profile_ngram_commit.py` 参数：
  - `--kv-offload-mvp0`
  - `--kv-offload-gpu-blocks N`
  - `--kv-offload-logical-blocks N`

实现范围：

- 仅支持 fp16/bf16 KV（`kv_quant_bits == 0`）。
- 仅支持 full attention；启用时会拒绝 Quest / KV-Cartridge / AM compact。
- 强制走 eager decode / 跳过 cuda graph。
- `seq.block_table` 保持 logical block id。
- `ModelRunner` 内新增 `KVOffloadMVP0`：
  - GPU KV cache 按 `kv_offload_gpu_blocks` 分配为 staging slots。
  - CPU pinned KV backing store 按 `kv_offload_logical_blocks` 分配。
  - `prepare_prefill()` / `prepare_decode()` 在上传 metadata 前做 logical block id -> physical GPU slot id 翻译。
  - 每次 forward 后把写过的 logical block 从 GPU slot writeback 到 CPU pinned store。
  - `tools/profile_ngram_commit.py` 的 target verify 手工 context 路径也接入了同一套 remap/writeback。

本地已通过：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/config.py \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/engine/model_runner.py \
  /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py --help >/dev/null
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/test_chunked_prefill.py
```

远程已同步并通过：

```bash
cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
PYTHONDONTWRITEBYTECODE=1 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile \
  tinyvllm/config.py tinyvllm/engine/model_runner.py tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py --help >/dev/null
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_chunked_prefill.py
```

远程 Qwen3-0.6B smoke：

- baseline JSON：`profile_out/ngram_spec_s4_06b_baseline_kvoffload_mvp0_smoke_20260630.json`
- candidate JSON：`profile_out/ngram_spec_s4_06b_candidate_kvoffload_mvp0_smoke_20260630.json`

baseline-only 参数：`max_output_len=8`，`max_model_len=2048`，`kv_offload_gpu_blocks=16`，`kv_offload_logical_blocks=32`。

| mode | gate_pass | output_tokens | decode_steps | kv d2h copies | kv d2h MB | kv h2d copies |
|---|---:|---:|---:|---:|---:|---:|
| baseline-only | true | 8 | 7 | 8 | 234.9 | 0 |

candidate-only 参数：`max_output_len=16`，`ngram_size=3`，`max_draft_tokens=4`，`max_commit_events=0`，`max_model_len=2048`，`kv_offload_gpu_blocks=16`，`kv_offload_logical_blocks=32`。

| mode | gate_pass | output_tokens | decode_steps | commit_events | accepted | drafted | kv d2h copies | kv d2h MB | kv h2d copies |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate-only | true | 16 | 13 | 1 | 2 | 4 | 15 | 440.4 | 0 |

注意：这只是 MVP-0 正确性闭环。由于 smoke prompt 只占 1 个 KV block，未触发 eviction/H2D reload，所以 `h2d_copies=0` 是预期现象。下一步若要验证 page migration，需要构造“active full-attention blocks <= gpu staging slots，但跨请求/跨阶段会驱逐再读回”的用例，或进入 MVP-1 做 prefetch/eviction policy。

## 2026-06-30 KV offload MVP-0 page migration smoke

为了先验证真实 page migration 语义，而不是直接改复杂 scheduler，已给 `tools/profile_ngram_commit.py` 增加 synthetic smoke：

```bash
--kv-offload-migration-smoke
```

这个 smoke 不加载模型，直接构造一个小型 `KVOffloadMVP0`：

1. GPU staging slots = 2，logical blocks = 4。
2. 写 logical block 0/1 并 D2H writeback 到 CPU pinned backing store。
3. 访问 logical block 2/3，驱逐 0/1，并写回 2/3。
4. 再访问 logical block 0/1，触发从 CPU pinned backing store H2D reload。
5. 校验 reload 后 GPU slot 内容和原始写入内容一致。

本地已通过：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/engine/model_runner.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py --help >/dev/null
```

远程已同步并通过静态检查。远程 smoke：

```bash
cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 \
  /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --kv-offload-migration-smoke \
  --out-json profile_out/kv_offload_mvp0_page_migration_smoke_20260630.json
```

结果：

| metric | value |
|---|---:|
| gate_pass | true |
| evictions | 4 |
| h2d_copies | 2 |
| d2h_copies | 4 |
| h2d_bytes | 64 |
| d2h_bytes | 128 |
| resident_blocks | 2 |
| dirty_blocks | 0 |

输出文件：`profile_out/kv_offload_mvp0_page_migration_smoke_20260630.json`

结论：MVP-0 的核心 page migration 语义已打通：GPU staging slot 可驱逐，dirty logical block 可写回 CPU pinned store，之后可 H2D reload，并且 reload 内容校验通过。下一步如果继续做 MVP-1，应把这个同步迁移原语升级为：prefetch plan、LRU/成本感知 eviction、异步 copy stream/event、批量 block copy，以及 profiler 中的真实模型长上下文 thrash 场景。

## 2026-06-30 KV offload MVP-1

在 MVP-0 基础上做了保守的 MVP-1 增强，仍保持默认关闭，不改 scheduler / attention kernel / BlockManager 语义：

- `Config` 新增：
  - `kv_offload_async_copy=True`
  - `kv_offload_batch_copy=True`
  - `kv_offload_writeback_on_evict=False`
  - `kv_offload_evict_policy="lru_cost"`
- `KVOffloadMVP0` 增强：
  - 独立 CUDA copy stream + event。
  - H2D/D2H 不再每块全局 `torch.cuda.synchronize()`；forward 前只等待 required H2D events。
  - 连续 `(logical_block, gpu_slot)` span 合并成批量 copy。
  - LRU + dirty/future reuse penalty 的成本感知 victim 选择。
  - stats 增加：`copy_waits`、`h2d_batches`、`d2h_batches`、`h2d_batch_spans`、`d2h_batch_spans`、`evict_clean`、`evict_dirty`、`prefetch_plans`、`prefetch_read_blocks`、`prefetch_write_blocks`。
- `ModelRunner.prepare_prefill()` / `prepare_decode()` 增加显式 prefetch plan 统计，仍把 attention metadata 翻译成 physical GPU slot id。
- `tools/profile_ngram_commit.py` 的 target verify 手工 context 路径同步接入 `_kv_offload_before_forward()` 和 async writeback。
- 新增 synthetic thrash smoke：
  - `--kv-offload-thrash-smoke`
  - `--thrash-gpu-blocks`
  - `--thrash-logical-blocks`
  - `--thrash-window-blocks`
  - `--thrash-rounds`
  - 以及调试开关 `--kv-offload-no-async-copy`、`--kv-offload-no-batch-copy`、`--kv-offload-writeback-on-evict`、`--kv-offload-evict-policy {lru,lru_cost}`。

本地已通过：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/config.py \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/engine/model_runner.py \
  /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py --help >/dev/null
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/test_chunked_prefill.py
```

远程已同步并通过同样的静态检查和纯 Python 回归。

### MVP-1 migration 回归 smoke

输出：`profile_out/kv_offload_mvp1_migration_regression_20260630.json`

| metric | value |
|---|---:|
| gate_pass | true |
| evictions | 4 |
| h2d_copies | 2 |
| d2h_copies | 4 |
| h2d_batches | 1 |
| d2h_batches | 2 |
| h2d_batch_spans | 1 |
| d2h_batch_spans | 2 |
| copy_waits | 6 |

### MVP-1 synthetic thrash smoke

命令口径：`thrash_gpu_blocks=3`，`thrash_logical_blocks=8`，`thrash_window_blocks=2`，`thrash_rounds=24`。

输出：`profile_out/kv_offload_mvp1_thrash_smoke_20260630.json`

| metric | value |
|---|---:|
| gate_pass | true |
| evictions | 53 |
| h2d_copies | 48 |
| d2h_copies | 8 |
| h2d_batches | 24 |
| d2h_batches | 3 |
| h2d_batch_spans | 32 |
| d2h_batch_spans | 3 |
| prefetch_plans | 24 |
| prefetch_read_blocks | 48 |

### 真实模型 two-request thrash smoke

短 prompt 两请求，`max_num_seqs=1`，`kv_offload_gpu_blocks=1`，用于验证真实 LLM 路径中两个 request 交替 decode 触发 staging reload。

输出：`profile_out/kv_offload_mvp1_real_two_request_thrash_20260630.json`

| metric | value |
|---|---:|
| gate_pass | true |
| output_tokens | 8 |
| evictions | 3 |
| h2d_copies | 2 |
| d2h_copies | 8 |
| h2d_batches | 2 |
| d2h_batches | 8 |
| prefetch_plans | 8 |

### 真实模型长上下文 thrash smoke

两条约 385-token prompt，`max_num_seqs=1`，`kv_offload_gpu_blocks=2`，`kv_offload_logical_blocks=16`。每条 request 当前 full-attention 可见 blocks <= staging slots，但两个 request 交替时会驱逐/重载。

输出：`profile_out/kv_offload_mvp1_real_longctx_thrash_20260630.json`

| metric | value |
|---|---:|
| gate_pass | true |
| output_tokens | 4 |
| evictions | 6 |
| h2d_copies | 4 |
| d2h_copies | 6 |
| h2d_batches | 2 |
| d2h_batches | 4 |
| prefetch_plans | 4 |
| prefetch_read_blocks | 4 |
| prefetch_write_blocks | 6 |

结论：MVP-1 的 prefetch plan / cost-aware LRU / async stream-event / batch copy / synthetic 和真实模型 thrash 场景都已打通。仍然保留 full attention 限制：单次 forward 的可见 logical blocks 必须 `<= kv_offload_gpu_blocks`；要支持单条超长上下文超过 staging slots，需要下一阶段做 streaming/blockwise attention，而不是仅靠 page migration。

### 2026-07-08 KV offload dirty eviction D2H batching

本轮继续性能主线，把 `ensure_resident()` 同一轮加载多个新 logical blocks 时触发的 dirty evictions 从逐 slot D2H 改成 deferred batched D2H：

- `_evict_slot(..., defer_dirty_writeback=True)` 只选择 victim 并统计 eviction，不立即 copy/wait/清映射。
- `ensure_resident()` 收集本轮被驱逐的 dirty `(logical_block, slot)`，统一调用一次 `_enqueue_d2h_pairs()`；连续 logical/slot pair 仍由 `_coalesce_copy_pairs()` 合并 span。
- clean eviction 如果已有 pending D2H event，也会在复用 slot 前等待，避免复用还在写回的 GPU slot。

新增 `tools/test_kv_offload.py`：

- `test_dirty_evictions_are_batched_when_loading_multiple_blocks()`：2 个 dirty resident blocks 被同一轮 `[2,3]` 加载驱逐时，要求 `d2h_copies=2`、`d2h_batches=1`、`d2h_batch_spans=1`。
- `test_deferred_clean_eviction_waits_for_pending_d2h_event()`：先手动 `writeback_dirty([0,1])`，再驱逐 clean blocks，要求 `copy_waits` 至少增加 2，证明复用 slot 前等待 pending D2H event。

验证：

- 本地：`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/engine/model_runner.py tools/test_kv_offload.py`、`git diff --check` 通过。
- 远程 GPU4：`CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_kv_offload.py` 通过。
- 远程 migration smoke：`profile_out/kv_offload_batched_dirty_evict_migration_20260708_r2.json`，`gate_pass=true`、`h2d_copies=2`、`d2h_copies=4`、`h2d_batches=1`、`d2h_batches=2`、`d2h_batch_spans=2`、`copy_waits=6`。
- 远程 thrash smoke：`profile_out/kv_offload_batched_dirty_evict_thrash_20260708_r2.json`，`gate_pass=true`、`h2d_copies=8`、`d2h_copies=6`、`h2d_batches=4`、`d2h_batches=2`、`d2h_batch_spans=2`、`prefetch_plans=4`。

### 2026-07-08 KV offload copy event wait coalescing

继续减少 copy/wait 调度开销：batched H2D/D2H 会给多个 logical blocks 记录同一个 CUDA event，旧 `wait_for_blocks()` 和 deferred eviction wait 会对同一个 event 重复 `wait_event()` 并重复累计 `copy_waits`。

- `wait_for_blocks()` 现在按 `id(event)` 去重，同一个 H2D event 只 wait/统计一次。
- `ensure_resident()` 的 deferred D2H wait 同样按 event 去重；clean eviction 仍会在复用 slot 前等待 pending D2H，但同一 batched D2H event 只等待一次。
- 新增测试：
  - `test_wait_for_blocks_coalesces_identical_h2d_events()`
  - `test_deferred_eviction_waits_once_per_identical_d2h_event()`
- 本地：`py_compile tinyvllm/engine/model_runner.py tools/test_kv_offload.py`、`git diff --check` 通过。
- 远程 GPU4：`tools/test_kv_offload.py` 通过。
- 远程 migration smoke：`profile_out/kv_offload_wait_coalesce_migration_20260708.json`，`gate_pass=true`、`copy_waits=4`（上一轮同口径 dirty eviction batching 后为 6）。
- 远程 thrash smoke：`profile_out/kv_offload_wait_coalesce_thrash_20260708.json`，`gate_pass=true`、`copy_waits=10`（上一轮同口径为 19）、`h2d_batches=4`、`d2h_batches=2`、`prefetch_plans=4`。

### 2026-07-08 KV offload pending wait cleanup

继续消除重复 wait：`ensure_resident(..., wait=True)` 已同步等待本轮 H2D blocks，但旧逻辑仍把这些 logical blocks 留在 `pending_wait_blocks`，后续 `wait_for_pending()` 会再次等待同一批 H2D event。

- `ensure_resident(wait=True)` 现在在 `wait_for_blocks()` 后从 `pending_wait_blocks` 移除已等待的 H2D blocks。
- 新增 `test_ensure_resident_wait_clears_pending_h2d_waits()`：要求 wait=True 后 `{0,1}` 不再残留 pending，并且后续 `wait_for_pending()` 不增加 `copy_waits`。
- 本地 `py_compile tinyvllm/engine/model_runner.py tools/test_kv_offload.py`、`git diff --check` 通过。
- 远程 GPU4 `tools/test_kv_offload.py` 通过。
- 远程 migration smoke：`profile_out/kv_offload_pending_wait_clear_migration_20260708.json`，`gate_pass=true`、`copy_waits=4`。
- 远程 thrash smoke：`profile_out/kv_offload_pending_wait_clear_thrash_20260708.json`，`gate_pass=true`、`copy_waits=10`、`h2d_batches=4`、`d2h_batches=2`、`prefetch_plans=4`。

### 2026-07-08 KV offload pending H2D anti-thrash

继续降低 clean eviction 抖动：batched H2D 发起后，logical block 会留在 `pending_wait_blocks`。如果该 block 还没被 forward 等待/消费，就被 victim policy 选中驱逐，会浪费一次 in-flight H2D copy。`lru_cost` 的 victim score 现在对 `logical_block in pending_wait_blocks` 增加 `pending_h2d_penalty = block_nbytes * 6.0`，在有其他候选时避免驱逐刚发起 H2D 的 block。

- 新增 `test_evict_policy_avoids_pending_h2d_block_when_possible()`，手动构造 pending H2D block 是最老 slot 的场景，要求仍保留 pending block、优先驱逐非 pending clean block。
- 本地 `py_compile tinyvllm/engine/model_runner.py tools/test_kv_offload.py`、`git diff --check` 通过。
- 远程 GPU4 `tools/test_kv_offload.py` 通过。
- 远程 thrash smoke：`profile_out/kv_offload_pending_h2d_penalty_thrash_20260708.json`，`gate_pass=true`、`copy_waits=10`、`h2d_copies=8`、`d2h_copies=6`、`evictions=11`、`prefetch_plans=4`。

### 2026-07-08 Blockwise attention window-specific H2D waits

继续降低 blockwise decode/prefill 的窗口级同步面：旧 `_blockwise_online_decode_attention()` 和 `_blockwise_online_prefill_attention()` 每个 read window 都调用 `manager.wait_for_pending()`，会 drain 全局 `pending_wait_blocks`。如果 prepare 阶段或其他窗口已经发起了尚未被当前窗口消费的 H2D，这会提前等待无关 copy，削弱 copy/compute overlap。

- `KVOffloadMVP0.wait_for_blocks()` 新增 `clear_pending=False` 参数；调用方可只等待指定 logical blocks，并只从 `pending_wait_blocks` 移除这些 blocks。
- `wait_for_pending()` 仍保留 drain-all 语义，内部复用 `wait_for_blocks(..., clear_pending=True)`。
- blockwise decode/prefill read window 现在改成 `manager.wait_for_blocks(list(unique_blocks), clear_pending=True)`，只等待当前窗口真正要读的 H2D。
- 新增测试：
  - `test_wait_for_blocks_clear_pending_api_without_cuda()`：用 fake stream 覆盖 API 语义，证明只清理请求的 pending blocks。
  - `test_wait_for_blocks_can_clear_only_requested_pending_h2d_waits()`：CUDA 场景下要求 `[0]` 被等待/清理后，未请求的 pending block `[1]` 仍保留。
- TDD RED：远程临时回退旧 `wait_for_blocks(self, logical_blocks)` 签名后，新增测试按预期失败：`TypeError: KVOffloadMVP0.wait_for_blocks() got an unexpected keyword argument 'clear_pending'`。
- 本地：`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/test_kv_offload.py`、`git diff --check` 通过。
- 远程 GPU4：`CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_kv_offload.py` 通过。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_window_wait_clear_20260708.json`，`gate_pass=true`、`chunks=8`、`streamed_tokens=1024`、`max_abs_error=2.9802322387695312e-08`、`relative_error=1.853052561279985e-07`。
  - `profile_out/blockwise_prefill_window_wait_clear_20260708.json`，`gate_pass=true`、`chunks=36`、`streamed_tokens=4544`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。
- 远程真实模型 smoke：`SMOKE_TAG=20260708_window_wait_clear RUN_PREFLIGHT=0 CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34687 MASTER_PORT=34687 tools/smoke_blockwise_prefill_remote.sh` 通过；真实长 prompt 输出 `profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260708_window_wait_clear.json`，`gate_pass=true`、`elapsed_s=29.39703532680869`、`output_tokens=1`。

### 2026-07-08 Blockwise attention ordered read windows

继续降低 blockwise window planning 的不确定性：旧逻辑先把 window blocks 放进 `set`，再 `list(unique_blocks)` 传给 `ensure_resident()` / `wait_for_blocks()`。这会丢掉跨 batch row 的 first-seen 顺序，使 H2D request / LRU touch / victim 决策顺序依赖 set 迭代，后续更难稳定优化 prefetch plan。

- 新增 `_unique_blocks_in_order()`，对 read window 逻辑块做 first-seen 去重，保留窗口内访问顺序。
- blockwise decode/prefill read window 都改成用 `unique_block_list` 调 `ensure_resident()` 和 `wait_for_blocks()`；`unique_blocks` 只保留给容量检查和 future/protected set。
- 新增 `tools/test_blockwise_attention_planning.py`，用 fake manager 验证 decode window `[2,0] + [1,2]` 的 staging/wait 顺序必须是 `[2,0,1]`。
- TDD RED：旧 set 路径下远程测试失败于 `assert manager.ensure_calls == [[2, 0, 1]]`。
- 本地：`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/layers/attention.py tools/test_blockwise_attention_planning.py`、`git diff --check` 通过。
- 远程：`tools/test_blockwise_attention_planning.py` 通过，`tools/test_kv_offload.py` 通过。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_ordered_windows_20260708.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`、`relative_error=1.853052561279985e-07`。
  - `profile_out/blockwise_prefill_ordered_windows_20260708.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。

## 2026-06-30 Streaming/blockwise attention 数学 smoke

为了进入“单条超长上下文超过 staging slots”的下一阶段，先没有直接改 production attention kernel，而是在 `tools/profile_ngram_commit.py` 增加了 exact blockwise decode attention 的 online-softmax smoke：

```bash
--blockwise-attn-smoke
--blockwise-attn-batch N
--blockwise-attn-heads N
--blockwise-attn-kv-heads N
--blockwise-attn-head-dim N
--blockwise-attn-tokens N
--blockwise-attn-window-tokens N
```

实现口径：

- 构造 synthetic decode `q/k/v`。
- full attention 一次性计算作为 reference。
- blockwise 路径按 `window_tokens` 分块扫描 K/V。
- 使用 online softmax merge：维护 running max、running denominator、running output。
- 支持 GQA：`num_heads` 可大于 `num_kv_heads`，通过 KV head repeat 做对齐。
- 校验 blockwise 输出与 full attention 输出误差。

本地通过：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py --help >/dev/null
```

注意：本地无 torch，实际 smoke 在远程跑。

远程 smoke：

```bash
cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 \
  /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --blockwise-attn-smoke \
  --blockwise-attn-tokens 2048 \
  --blockwise-attn-window-tokens 128 \
  --blockwise-attn-batch 2 \
  --blockwise-attn-heads 4 \
  --blockwise-attn-kv-heads 2 \
  --blockwise-attn-head-dim 32 \
  --out-json profile_out/blockwise_attn_online_softmax_smoke_20260630.json
```

结果：

| metric | value |
|---|---:|
| gate_pass | true |
| chunks | 16 |
| streamed_tokens | 2048 |
| max_abs_error | 2.98e-08 |
| relative_error | 2.31e-07 |
| context_lens | [2048, 1984] |

输出文件：`profile_out/blockwise_attn_online_softmax_smoke_20260630.json`

结论：exact blockwise/streaming decode attention 的数学闭环已验证；下一步若接真实 KV offload，需要把这个 online-softmax merge 放到真实 decode attention 路径，并让每个 block/window 触发 KV staging/prefetch，而不是要求一次性把所有 visible blocks staged 到 GPU。

## 2026-06-30 真实 decode attention 接入 blockwise KV offload

已把 blockwise online-softmax merge 接进真实 decode attention 路径，仍然默认关闭、仅覆盖受控范围：

- 仅 `kv_offload_mvp0=True` 时可用。
- 仅 fp16/bf16 KV（`kv_quant_bits == 0`）。
- 仅 decode path；prefill 仍走原路径。
- 不支持 Quest / KV-Cartridge / AM compact / KV4 / KV8。
- 不改 scheduler / BlockManager / attention kernel；blockwise decode MVP 使用 PyTorch gather + online-softmax，正确性优先，性能不是最终形态。

新增配置 / CLI：

```bash
--kv-offload-blockwise-decode
--kv-offload-blockwise-blocks N
--max-num-prefill-tokens-per-step N
```

关键实现：

- `tinyvllm/config.py`
  - `kv_offload_blockwise_decode`
  - `kv_offload_blockwise_blocks`
- `tinyvllm/utils/context.py`
  - 在 context 中传入 `kv_offload_manager`、logical block tables、context lens、write blocks、blockwise window size。
- `tinyvllm/engine/model_runner.py`
  - `prepare_decode()` 在 blockwise 模式下只预先 stage 当前 write blocks。
  - 不再要求 decode visible logical blocks 一次性全部 resident。
  - logical block rows 传给 attention，由 attention layer 按 window 触发 staging。
- `tinyvllm/layers/attention.py`
  - decode 时若 `context.kv_offload_blockwise_decode=True`：
    - 当前 token KV 先写入 staging slot。
    - 标记 write block dirty，避免窗口 staging 时丢当前层 KV。
    - 按 logical block window 调 `KVOffloadMVP0.ensure_resident()`。
    - 从当前 layer 的 physical slot gather window K/V。
    - 用 online softmax merge 每个 window 的 attention 结果。

本地已通过：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/config.py \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/utils/context.py \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/layers/attention.py \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/engine/model_runner.py \
  /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py --help >/dev/null
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/test_chunked_prefill.py
```

远程已同步并通过静态检查。

### blockwise attention 数学回归

输出：`profile_out/blockwise_attn_online_softmax_regression_20260630.json`

| metric | value |
|---|---:|
| gate_pass | true |
| chunks | 16 |
| streamed_tokens | 2048 |
| max_abs_error | 2.98e-08 |
| relative_error | 2.31e-07 |

### 真实模型单请求 blockwise decode smoke

Qwen3-0.6B，单条约 385-token prompt，`kv_offload_gpu_blocks=2`，`kv_offload_blockwise_blocks=1`。

输出：`profile_out/kv_offload_blockwise_decode_real_smoke_20260630.json`

| metric | value |
|---|---:|
| gate_pass | true |
| output_tokens | 2 |
| decode_steps | 1 |
| prefetch_plans | 58 |
| prefetch_read_blocks | 56 |
| prefetch_write_blocks | 3 |
| d2h_copies | 3 |
| h2d_copies | 0 |

这里没有 H2D reload 是预期的：单请求 2 个 logical blocks 正好等于 staging slots，未触发驱逐。

### 真实模型两请求 blockwise decode thrash

两条约 385-token prompt，`max_num_seqs=1`，`kv_offload_gpu_blocks=2`，`kv_offload_blockwise_blocks=1`，两个 request 交替触发 staging eviction/reload。

输出：`profile_out/kv_offload_blockwise_decode_real_two_request_thrash_20260630.json`

| metric | value |
|---|---:|
| gate_pass | true |
| output_tokens | 4 |
| decode_steps | 2 |
| evictions | 6 |
| h2d_copies | 4 |
| d2h_copies | 6 |
| h2d_batches | 4 |
| d2h_batches | 4 |
| prefetch_plans | 116 |
| prefetch_read_blocks | 112 |
| prefetch_write_blocks | 6 |

当前限制 / 下一步：

- decode attention 已不再需要“一次性 stage 全部 visible logical blocks”；它按 window stage。
- 已新增 blockwise/chunked prefill correctness path：`--kv-offload-blockwise-prefill` 下，chunked prefill 的 prefix read blocks 不再在 `prepare_prefill()` 一次性全量 resident，而是由 attention layer 按 window stage，并用 online softmax merge。
- 当前 blockwise prefill 仍要求 `prefix window blocks + 当前 chunk write blocks <= kv_offload_gpu_blocks`，因此需要通过 `--max-num-prefill-tokens-per-step` 控制当前 chunk 写入 blocks 数。
- 本地已通过 `py_compile`、`tools/test_chunked_prefill.py`、`tools/test_ngram_speculative.py`、`git diff --check`；本地无 torch，数学 smoke 和真实模型长 prompt smoke 待远程 GPU 运行。
- 当前 blockwise prefill/decode 都是 PyTorch correctness path，性能版需要 Triton/FlashAttention 风格的 window kernel 或能返回 logsumexp 的 flash 分块接口。

## 2026-07-02 Blockwise/chunked prefill 远程 smoke 已完成

本地 7 个改动文件已同步到远程运行目录 `/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge`，未同步 `.agents/`、`.codex/`、`needle_sq_results/`。

远程环境确认：

```text
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python
torch 2.4.1+cu121
torch.cuda.is_available() == True
torch.cuda.device_count() == 8
```

远程已通过：

```bash
PYTHONDONTWRITEBYTECODE=1 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile \
  tinyvllm/config.py tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py \
  tinyvllm/utils/context.py tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge \
  /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge \
  /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_ngram_speculative.py
```

结果：`chunked prefill tests passed`、`ngram speculative tests passed`。

数学 smoke：

```bash
cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 \
  /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --blockwise-prefill-attn-smoke \
  --blockwise-prefill-prefix-tokens 1024 \
  --blockwise-prefill-chunk-tokens 128 \
  --blockwise-attn-window-tokens 128 \
  --out-json profile_out/blockwise_prefill_attn_online_softmax_smoke_20260630.json
```

结果：

| metric | value |
|---|---:|
| gate_pass | true |
| chunks | 18 |
| streamed_tokens | 2240 |
| max_abs_error | 2.5331974029541016e-07 |
| relative_error | 1.0576243517384856e-06 |

真实模型长 prompt smoke：

```bash
CUDA_VISIBLE_DEVICES=7 TINYVLLM_DIST_PORT=34567 MASTER_PORT=34567 \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --mode baseline-only \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B \
  --prompt '<约 1381 tokens，超过 2 个 KV blocks 且不超过 8 个 logical blocks 的长 prompt>' \
  --max-output-len 1 \
  --temperature 0.0 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7 \
  --max-num-seqs 1 \
  --max-num-prefill-tokens-per-step 256 \
  --kv-offload-mvp0 \
  --kv-offload-gpu-blocks 2 \
  --kv-offload-logical-blocks 8 \
  --kv-offload-blockwise-prefill \
  --kv-offload-blockwise-decode \
  --kv-offload-blockwise-blocks 1 \
  --out-json profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260630.json
```

结果：

| metric | value |
|---|---:|
| gate_pass | true |
| output_tokens | 1 |
| decode_steps | 0 |
| elapsed_s | 36.93855146318674 |
| prefill chunks | 6 |
| chunk token sizes | 256, 256, 256, 256, 256, 101 |
| h2d_copies | 391 |
| d2h_copies | 6 |
| evictions | 395 |
| prefetch_plans | 426 |
| prefetch_read_blocks | 420 |
| prefetch_write_blocks | 6 |
| logical_blocks | 8 |
| gpu_blocks | 2 |
| resident_blocks | 2 |

注意：一次无效尝试使用了约 4141-token prompt，超过 `max_model_len=4096` 且超过 `kv_offload_logical_blocks=8` 可覆盖容量，触发 scheduler 空 decode assert；这不是 blockwise prefill correctness 失败。另一次复跑遇到端口占用，后续显式设置 `TINYVLLM_DIST_PORT=34567 MASTER_PORT=34567` 后通过。

后续改进机会与当前落地状态：

1. 已落地：在 `Scheduler.add()` 增加 admission 前置校验，提前拒绝 `prompt_tokens + max_tokens > max_model_len`，以及最大 KV footprint 超过 logical KV blocks 的请求，避免无效 prompt 后续退化成 scheduler 空 decode assert。
2. 已落地：在 `tools/test_chunked_prefill.py` 增加 max_model_len、prompt 超 logical KV 容量、decode KV 容量边界测试，并补充 `num_kvcache_blocks=1/2/4`、`max_num_prefill_tokens_per_step=1/2/4`、短 prompt batch 受 `max_num_seqs/max_num_batched_tokens` 限制的本地系统边界测试。
   - 2026-07-03 本地验证：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge python3 tools/test_chunked_prefill.py` 通过；`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tools/test_chunked_prefill.py tinyvllm/engine/scheduler.py` 通过；`git diff --check` 通过。
3. 已落地：新增 `tools/smoke_blockwise_prefill_remote.sh`，默认封装远程 Python、`CUDA_VISIBLE_DEVICES=7`、`TINYVLLM_DIST_PORT=34567`、`MASTER_PORT=34567`、`PYTHONPATH`、数学 smoke 和真实模型长 prompt smoke；远程运行目录下直接执行 `tools/smoke_blockwise_prefill_remote.sh` 即可，常用覆盖如 `RUN_REAL_SMOKE=0`、`CUDA_VISIBLE_DEVICES=0`、`MODEL_PATH=/path/to/model`。
   - 2026-07-03 已在远程用 `SMOKE_TAG=20260703_script tools/smoke_blockwise_prefill_remote.sh` 完整验证入口：preflight 通过，数学 smoke `gate_pass=true`，真实模型长 prompt smoke `gate_pass=true`。
   - 输出：`profile_out/blockwise_prefill_attn_online_softmax_smoke_20260703_script.json`、`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260703_script.json`；对应 log 写到同名 `.log`，避免长 prompt JSON 全量刷终端。
   - 2026-07-06 已扩展轻量 GPU blocks matrix 模式：`RUN_GPU_BLOCKS_MATRIX=1`，默认 `KV_OFFLOAD_GPU_BLOCKS_MATRIX="1 2 4"`，每组输出带 `_gpuN` 后缀的 JSON/log，并汇总 `gate_pass`、`elapsed_s`、`h2d_copies`、`d2h_copies`、`evictions`、`resident_blocks`。
4. 远程/GPU 单 prompt 矩阵已验证：`SMOKE_TAG=20260706_gpu_matrix RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 RUN_GPU_BLOCKS_MATRIX=1 MATRIX_REQUIRE_PASS=0 tools/smoke_blockwise_prefill_remote.sh`。
   - `gpu_blocks=1`：预期容量失败，log 中报 `blockwise prefill window plus current write blocks exceed GPU staging slots: required=2, gpu_blocks=1`；这是当前实现边界，不是 correctness mismatch。
   - `gpu_blocks=2`：`gate_pass=true`，`elapsed_s=60.20214011892676`，`h2d_copies=391`，`d2h_copies=6`，`evictions=395`，`resident_blocks=2`。
   - `gpu_blocks=4`：`gate_pass=true`，`elapsed_s=55.02894039079547`，`h2d_copies=249`，`d2h_copies=6`，`evictions=251`，`resident_blocks=4`。
   - 2026-07-06 已补多 prompt batch smoke：`SMOKE_TAG=20260706_multiprompt_len2048 RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 RUN_REAL_SMOKE=0 RUN_MULTI_PROMPT_SMOKE=1 MAX_MODEL_LEN=2048 GPU_MEMORY_UTILIZATION=0.85 KV_OFFLOAD_LOGICAL_BLOCKS=8 MULTI_PROMPT_REPEAT=24 tools/smoke_blockwise_prefill_remote.sh`，`num_prompts=2`、`gate_pass=true`、`elapsed_s=32.339650828391314`、`output_tokens=2`、`h2d_copies=786`、`d2h_copies=12`、`evictions=792`、`resident_blocks=2`。
5. 待做：性能路径优化，包括合并 H2D/D2H copy、合并连续 prefetch plan、降低 clean eviction 抖动，以及后续引入 Triton/FlashAttention 风格 window kernel。
6. 待做：进一步抽象统一的 KV block access planner，让 prefill/decode 共享 `plan_read_blocks()`、`stage_blocks()`、`evict_blocks()`、`commit_write_blocks()` 语义。
7. 已落地：DFlash feasibility spike 已写入 `docs/dflash-feasibility.md`，只做接口/接入点预研，不直接实现完整 DFlash；已完成 Phase 1，把 n-gram target verify/commit 包装为通用 `verify_and_commit_block()` 并保留 n-gram 行为不变。远程 Qwen3-0.6B candidate-only smoke 已验证 `commit_event.draft_source="ngram"`。Phase 2 plumbing 已验证：新增 `--draft-source {ngram,dflash-toy,dflash-toy-ngram-or-repeat}`、`--allow-zero-accept`、deterministic `repeat_recent_tokens` toy block draft model，以及记录 zero-accept attempts 的 `verify_events`；远程 `dflash-toy` smoke `gate_pass=true`、`zero_accept_events=3`、`verify_events[].draft_source="dflash-toy"`。
   - 2026-07-07 accepted-friendly toy smoke 已通过：`--draft-source dflash-toy-ngram-or-repeat` 输出 `profile_out/dflash_phase2_toy_hybrid_candidate_smoke_20260707.json`，`gate_pass=true`、`commit_events=1`、`accepted_count=2`、`acceptance_rate=1.0`、`draft_metadata.toy_strategy="ngram_or_repeat"`、`draft_metadata.selected_strategy="ngram"`、`match_start=7`、`ngram_size=3`。
   - 2026-07-07 profiler-only hidden debug 已通过：`--debug-target-hidden` 输出 `profile_out/dflash_phase2_hidden_debug_smoke_20260707.json`，`gate_pass=true`、`accepted_count=2`、`target_hidden_debug.shape=[3, 1024]`、`target_hidden_debug.dtype="torch.bfloat16"`、`target_hidden_debug.device="cuda:0"`。该路径只在 profiler verify hook 调 `run_model(..., return_hidden=True)`，不改 `LLMEngine.step()` 或核心 runtime。下一步若继续 DFlash，应先做真实 draft model stub / hidden-to-draft adapter 的 profiler-only 实验。
   - 2026-07-07 hidden-to-draft adapter stub 已通过：`--debug-hidden-to-draft-stub --debug-hidden-to-draft-top-k 2` 输出 `profile_out/dflash_phase3_hidden_to_draft_stub_smoke_20260707.json`，`gate_pass=true`、`accepted_count=2`、`hidden_to_draft_stub.adapter="target_hidden_topk_stub"`、`shape=[3, 1024]`、`dtype="torch.bfloat16"`、`device="cuda:0"`、`top_k=2`、`rows=2`；preview 中 row 0 top tokens `[13440, 8287]`，row 1 top tokens `[21619, 13440]`。stub 只记录 target hidden metadata 和 verify logits top-k preview，不采样、不替换 draft tokens、不改 acceptance rule/runtime。
   - 2026-07-07 hidden-to-draft adapter interface schema/timing 已通过：输出 `profile_out/dflash_phase3_adapter_interface_smoke_20260707.json`，`gate_pass=true`、`accepted_count=2`、`hidden_to_draft_stub.interface_version=1`、`runtime_mutation=false`、`input_schema.hidden_states.shape=[3, 1024]`、`input_schema.logits.shape=[2, 151936]`、`output.draft_token_ids=[13440, 21619]`、`timing_ms.adapter_total_ms=142.16194674372673`、`logits_to_cpu_ms=8.412063121795654`、`topk_ms=133.64192843437195`。当前 timing 是 Python profiler preview 成本，不代表未来优化 adapter latency；价值是固定 profiler-only ABI 和计时字段。
   - 2026-07-07 `--hidden-to-draft-adapter {topk-stub,linear-stub}` 和 `linear-stub` skeleton 已通过：第一次 `TINYVLLM_DIST_PORT=34574 MASTER_PORT=34574` 失败于 `EADDRINUSE`，换 `34674` 重跑成功；输出 `profile_out/dflash_phase3_linear_stub_interface_smoke_20260707.json`，`gate_pass=true`、`accepted_count=2`、`hidden_to_draft_adapter="linear-stub"`、`hidden_to_draft_stub.adapter="target_hidden_linear_stub"`、`input_schema.adapter="linear-stub"`、`output_schema.projection="deterministic_placeholder"`、`output.draft_token_ids=[13440, 21619]`、`timing_ms.linear_projection_ms=0.0002868473529815674`。`linear-stub` 当前只改变 profiler JSON 和 no-op projection timing，不参与 acceptance/runtime。
   - 2026-07-07 `linear-stub` deterministic hidden projection skeleton 已通过：把 no-op projection 换成固定 seed/pseudo weight 的 hidden rows -> 小 vocab candidate set 投影，仍只写 profiler JSON，不参与 draft proposal、acceptance、commit 或 runtime。第一次 `TINYVLLM_DIST_PORT=34676 MASTER_PORT=34676` 失败于 `EADDRINUSE`，换 `35676` 重跑成功；输出 `profile_out/dflash_phase3_linear_projection_stub_smoke_20260707_r2.json`，`gate_pass=true`、`accepted_count=2`、`hidden_to_draft_stub.adapter="target_hidden_linear_stub"`、`output_schema.projection="deterministic_hidden_linear_stub"`、`projection_metadata.seed=17`、`candidate_token_ids=[0,1,2,3,4,5,6,7]`、`hidden_dim=1024`、`candidate_count=8`、`output.draft_token_ids=[0,0,7]`、`rows=3`、`output.num_rows=3`、`input_schema.hidden_states.shape=[3,1024]`、`input_schema.logits.shape=[3,151936]`、`timing_ms.hidden_to_cpu_ms=0.17523393034934998`、`linear_projection_ms=3.5200342535972595`、`adapter_total_ms=12.286126613616943`。已修复 hidden rows 与 logits rows 数量不一致时 `rows`/`output.num_rows` 误按 logits rows 计数的问题。
   - 2026-07-07 `topk-stub` vs `linear-stub` 3x remote compare 已通过：固定端口序列第二个进程仍遇到 `EADDRINUSE`，改为远程动态探测空闲端口后 6 个 JSON 均成功（`profile_out/dflash_phase3_adapter_compare_{topk-stub,linear-stub}_r{1,2,3}_20260707_cmp2.json`）。`topk-stub` 三次均 `gate_pass=true`、`output.rows=2`、`draft_token_ids=[13440,21619]`、`projection="logits_topk"`、`adapter_total_ms=124.903983±1.449532`、`topk_ms=116.005514±1.719655`；`linear-stub` 三次均 `gate_pass=true`、`output.rows=3`、`draft_token_ids=[0,0,7]`、`projection="deterministic_hidden_linear_stub"`、`adapter_total_ms=12.578132±0.081115`、`hidden_to_cpu_ms=0.191076±0.019228`、`linear_projection_ms=3.593331±0.057019`。ABI 字段足够继续 profiler-only 真实 draft model stub，但真实接入前建议显式化 `hidden_rows`/`logit_rows`/`projected_rows` 并拆出 `draft_model_forward_ms`、`candidate_select_ms`。
   - 2026-07-07 已继续补齐 ABI/timing 字段：`input_schema` 新增 `hidden_rows`、`logit_rows`、`projected_rows`，`output_schema`/`output` 新增 `projected_rows`，`timing_ms` 新增 `draft_model_forward_ms` 与 `candidate_select_ms`；同时修正 `linear-stub` 下 `input_schema.logits.shape` 始终按真实 logits rows 记录，不再被 hidden projected rows 覆盖。本地测试已覆盖 hidden rows=3、logit rows=2、projected rows=3 的不一致场景；远程 `profile_out/dflash_phase3_adapter_abi_fields_smoke_20260707.json` 验证 `gate_pass=true`、`accepted_count=2`、`hidden_rows=3`、`logit_rows=2`、`projected_rows=3`、`input_schema.logits.shape=[2,151936]`、`output.projected_rows=3`、`candidate_select_ms=3.570154309272766`、`draft_model_forward_ms=0.0`。
   - 2026-07-07 已新增 profiler-only `draft-model-stub` adapter：CLI 支持 `--hidden-to-draft-adapter draft-model-stub`，输出 `target_hidden_draft_model_stub`、`projection="deterministic_draft_model_stub"`、`output.candidate_token_ids`、`output.candidate_logits`、`draft_model_metadata`，并真实占用 `draft_model_forward_ms`；仍保持 `runtime_mutation=false`，不影响 draft proposal/acceptance/runtime。远程第一次用 GPU7 失败于 `assert auto_num_blocks > 0`，原因是 GPU7 显存占用约 69GiB/80GiB；改用 GPU3 后 `profile_out/dflash_phase3_draft_model_stub_smoke_20260707_gpu3.json` 通过，`gate_pass=true`、`accepted_count=2`、`hidden_rows=3`、`logit_rows=2`、`projected_rows=3`、`output.draft_token_ids=[7,7,7]`、第一行 candidate ids `[7,1]`、`draft_model_forward_ms=3.5831667482852936`、`candidate_select_ms=0.021237879991531372`。commit 的 `draft_tokens`/`accepted_tokens` 仍是 `[13440,21619]`，证明 stub 未参与 acceptance/runtime。
   - 2026-07-07 已补 3-way adapter compare：GPU4 上对 `topk-stub`、`linear-stub`、`draft-model-stub` 各跑 3 次，输出 `profile_out/dflash_phase3_adapter_3way_compare_{topk-stub,linear-stub,draft-model-stub}_r{1,2,3}_20260707_cmp3.json`。9 次均 `gate_pass=true`、`accepted_count=2`，commit `draft_tokens`/`accepted_tokens` 均为 `[13440,21619]`。`draft-model-stub` 三次 candidate ids/logits、`draft_token_ids=[7,7,7]`、metadata 完全稳定；timing 均值/标准差：`adapter_total_ms=13.104054±0.257300`、`draft_model_forward_ms=3.721040±0.024405`、`candidate_select_ms=0.021578±0.001956`、`hidden_to_cpu_ms=0.349854±0.030066`。对照：`topk-stub adapter_total_ms=131.355740±6.488973`，`linear-stub adapter_total_ms=12.964208±0.502438`。下一步若继续，建议把 deterministic pseudo forward 边界抽成独立函数/类，再替换真实 draft model forward，仍保持 profiler-only `runtime_mutation=false`。
   - 2026-07-08 已把 deterministic pseudo forward 抽成 `run_draft_model_stub(hidden_rows, candidate_token_ids, top_k)`，输出 `candidate_token_ids`、`candidate_logits`、`draft_token_ids`、`draft_scores`、`preview`、`metadata`、`timing_ms`；`summarize_hidden_to_draft_stub(..., adapter="draft-model-stub")` 复用该 helper，方便后续替换真实 draft model forward。远程 `profile_out/dflash_phase3_draft_model_stub_boundary_smoke_20260708.json` 通过：`gate_pass=true`、`accepted_count=2`、`draft_token_ids=[7,7,7]`、第一行 candidate ids `[7,1]`、`draft_model_forward_ms=3.5160109400749207`、`candidate_select_ms=0.015269964933395386`，commit `draft_tokens`/`accepted_tokens` 仍为 `[13440,21619]`。
   - 2026-07-08 已把 draft model boundary 包成 dataclass/config shell：新增 `DraftModelStubConfig(seed, stub_version)`、`DraftModelResult(...).to_dict()`，`run_draft_model_stub(..., config=None) -> DraftModelResult`，并显式校验 empty candidates 与 ragged hidden rows；这是未来真实 draft model forward 的 profiler-only 返回/错误边界，不接入 `LLMEngine.step()`。本地 `tools/test_ngram_speculative.py`、`py_compile`、`tools/test_chunked_prefill.py`、`git diff --check` 均通过；远程 `tools/test_ngram_speculative.py` 与 Qwen3 smoke `profile_out/dflash_phase3_draft_model_dataclass_shell_smoke_20260708.json` 通过，`gate_pass=true`、`accepted_count=2`、`runtime_mutation=false`、`input_schema.adapter="draft-model-stub"`、`output.draft_token_ids=[7,7,7]`、第一行 candidate ids `[7,1]`、`draft_model_metadata.stub_version=1`、`draft_model_forward_ms=4.899017512798309`、`candidate_select_ms=0.01652538776397705`；commit `draft_tokens`/`accepted_tokens` 仍为 `[13440,21619]`。
   - 2026-07-08 已继续补输入侧 contract：新增 `DraftModelInput.from_rows(...).to_dict()/schema()`，显式记录 `hidden_rows`、`candidate_token_ids`、`top_k`、`source_shape`、`source_dtype`、`source_device`；`run_draft_model_stub()` 保持旧签名兼容，也可直接接收 `DraftModelInput`。本地 `tools/test_ngram_speculative.py`、`py_compile`、`tools/test_chunked_prefill.py`、`git diff --check` 均通过；远程 `profile_out/dflash_phase3_draft_model_input_contract_smoke_20260708.json` 通过，`gate_pass=true`、`accepted_count=2`、`runtime_mutation=false`、`draft_model_metadata.input_schema.source_shape=[3,1024]`、`source_dtype="torch.bfloat16"`、`source_device="cuda:0"`、`top_k=2`、`output.draft_token_ids=[7,7,7]`、第一行 candidate ids `[7,1]`、`draft_model_forward_ms=5.126491189002991`、`candidate_select_ms=0.02197548747062683`；commit `draft_tokens`/`accepted_tokens` 仍为 `[13440,21619]`。下一步若继续 DFlash，建议做多 prompt/batch shape smoke 或把 profiler-only draft model schema 抽成小模块，仍不要接 runtime。
   - 2026-07-08 已完成多 prompt / batch shape smoke：远程 `profile_out/dflash_phase3_draft_model_batch_shape_smoke_20260708.json` 使用 2 个 prompt、`--max-num-seqs 2`、`draft-model-stub`、`top_k=2`，结果 `gate_pass=true`、`num_prompts=2`、`commit_events=2`、`accepted_count=4`。每个 prompt 各 `commit_events=1`、`verify_events=1`、`accepted_count=2`；两个 event 的 `draft_model_metadata.input_schema` 都独立记录 `hidden_rows=3`、`hidden_dim=1024`、`candidate_count=8`、`top_k=2`、`source_shape=[3,1024]`、`source_dtype="torch.bfloat16"`、`source_device="cuda:0"`。prompt0 commit tokens `[13440,21619]`、draft model ids `[7,7,7]`、第一行 candidate `[7,1]`；prompt1 commit tokens `[6303,6176]`、draft model ids `[1,2,2]`、第一行 candidate `[1,2]`。结论：batch 场景下 event 级 DraftModelInput schema 未发生 prompt 间串写，仍保持 `runtime_mutation=false`。下一步若继续 DFlash，建议把 profiler-only draft model schema 抽成小模块或补更多形状覆盖，仍不要接 runtime。
   - 2026-07-08 已把 profiler-only draft model schema 抽成 `tools/draft_model_schema.py` 小模块，包含 `DraftModelInput`、`DraftModelResult`、`DraftModelStubConfig`；`tools/profile_ngram_commit.py` 只导入这些 dataclass，测试直接加载该模块并确认 profiler 返回同一类对象。本地 `tools/test_ngram_speculative.py`、`py_compile tools/draft_model_schema.py tools/profile_ngram_commit.py tools/test_ngram_speculative.py`、`tools/test_chunked_prefill.py`、`git diff --check` 均通过；远程 `profile_out/dflash_phase3_draft_model_schema_module_smoke_20260708.json` 通过，`gate_pass=true`、`accepted_count=2`、`runtime_mutation=false`、`source_shape=[3,1024]`、`source_dtype="torch.bfloat16"`、`source_device="cuda:0"`，commit `draft_tokens`/`accepted_tokens=[13440,21619]` 不变。下一步若继续，建议补更多形状覆盖或做真实 draft model 接入前置检查（vocab/tokenizer/hidden_dim contract），仍不要接 runtime。
   - 2026-07-08 已补真实 draft model 接入前置 contract 检查：`tools/draft_model_schema.py` 新增 `DraftModelContract` 与 `validate_draft_model_contract()`，支持 profiler-only 校验 `expected_hidden_dim`、`target_vocab_size`、`draft_vocab_size`、`tokenizer_family` / `draft_tokenizer_family`，并记录 candidate token id min/max；本地测试覆盖 hidden_dim mismatch、candidate id 超 target vocab、tokenizer family mismatch。`run_draft_model_stub()` 默认记录宽松 `contract` metadata，也可传显式 contract；远程第一次 smoke 因 `EADDRINUSE` 失败，换动态高端口后 `profile_out/dflash_phase3_draft_model_contract_smoke_20260708.json` 通过：`gate_pass=true`、`accepted_count=2`、`runtime_mutation=false`、`contract.compatible=true`、`actual_hidden_dim=1024`、`candidate_id_min=0`、`candidate_id_max=7`，commit `draft_tokens`/`accepted_tokens=[13440,21619]` 不变。下一步若继续，建议补更多形状覆盖，仍不要接 runtime/真实 checkpoint。

2026-07-06 尝试记录 / 排坑记录：

2026-07-08 mixed scheduler token-budget reserve 已落地：

1. 改动：`Scheduler._schedule_chunked_prefill()` 增加内部 `max_prefill_tokens` 参数，`_schedule_mixed_prefill_decode()` 给 decode query row 预留并持续检查 `max_num_batched_tokens`，避免 mixed batch 实际 `prefill_tokens + decode_rows` 超过 token budget。
2. 新增测试：`test_mixed_prefill_reserves_token_budget_for_decode_queries()`、`test_mixed_decode_rows_respect_remaining_token_budget()`，覆盖 tight-budget 下 short prompt batching 与多 decode rows。
3. 本地验证通过：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD python3 tools/test_chunked_prefill.py`、`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD python3 tools/test_profile_chunked_prefill.py`、`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/engine/scheduler.py tinyvllm/engine/model_runner.py tinyvllm/engine/llm_engine.py tinyvllm/engine/sequence.py tinyvllm/config.py tools/profile_chunked_prefill.py tools/test_chunked_prefill.py`、`git diff --check`。
4. 远程纯测试通过：同步 `tinyvllm/engine/scheduler.py`、`tools/test_chunked_prefill.py` 后，远程 `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_chunked_prefill.py` 与 `tools/test_profile_chunked_prefill.py` 均通过。
5. GPU profiler smoke 暂未形成性能数字：远程 `tools/profile_chunked_prefill.py --mode mixed ... --max-num-batched-tokens 129` 启动后 log 为 0 字节，进程进入 D-state（`STAT=D`、`WCHAN=os_acquire_rwlock_write`），`kill -9` 后仍需等待内核态返回；同时 SSH 偶发 `Connection closed by UNKNOWN port 65535`。这次应记录为远程 GPU/driver/链路环境阻塞，不作为代码回归。
6. 文档：`docs/qwen3-8b-fixes.md` 已追加 `47.39.6 Mixed token-budget reserve`。

2026-07-08 mixed first-chunk budget clamp 已落地：

1. 问题：`max_num_prefill_tokens_per_step > mixed 剩余 token budget` 时，首个 prefill chunk 会过大，导致 decode row 无法追加，mixed 退化成普通 prefill。
2. 改动：`_schedule_chunked_prefill(max_prefill_tokens=...)` 把 token budget 传给首个 waiting seq、已有 `prefilling` seq 和后续 short prompt batching；`_schedule_one_prefill_chunk(..., max_chunk_tokens=...)` 按 `max_num_prefill_tokens_per_step`、`max_chunk_tokens`、剩余 prompt tokens 三者取最小值。
3. 新增测试：`test_mixed_first_prefill_chunk_shrinks_to_leave_decode_budget()`，覆盖 `max_num_batched_tokens=5`、`max_num_prefill_tokens_per_step=8` 下 prefill chunk 缩到 4 并保留 decode row。
4. 本地与远程纯测试均通过：`tools/test_chunked_prefill.py`、`tools/test_profile_chunked_prefill.py`；本地 `py_compile` 与 `git diff --check` 通过。
5. 文档：`docs/qwen3-8b-fixes.md` 已追加 `47.39.7 Mixed first-chunk budget clamp`。

1. 远程 SSH / Kerberos：
   - 裸 `ssh sitian@10.232.195.203` 在 TRAE 进程里曾报 `Connection closed by UNKNOWN port 65535`，根因不是目标机 22 端口不可达，而是当前进程看不到 macOS API Kerberos cache。
   - `nc -vz 10.232.195.203 22` 成功，说明目标端口可达；`klist` 在 TRAE 进程里报 `Cache not found: API:11111111-...`，而用户外部 Terminal 里 `klist` 显示 `sitian@BYTEDANCE.COM` 可用。
   - `jump-proxy-hl` 不允许普通 session，`ssh sitian@jump-proxy-hl 'echo jump-ok'` 会报 `unknown channel type: session`；正确验证方式是直连目标 `ssh sitian@10.232.195.203 'echo remote-ok'` 或用 `-W` 作为 ProxyCommand。
   - 已把 `~/.ssh/config` 的 `jump-proxy-hl` / `jump-proxy-hla` / `jump-proxy-lf` 用户显式改为 `sitian`，避免 ProxyCommand 默认用本机用户 `bytedance`。
   - 稳定方案：使用 file cache `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian` 建 ControlMaster，例如 `ssh -MNf -o ControlMaster=yes -o ControlPath=/tmp/ssh-sitian-10.232.195.203-new -o ControlPersist=2h sitian@10.232.195.203`；后续 rsync/ssh 使用 `ssh -S /tmp/ssh-sitian-10.232.195.203-new ...`。
   - 老 socket `/tmp/ssh-sitian-10.232.195.203` 偶发 `Connection closed by UNKNOWN port 65535`；遇到时重新建新 ControlMaster 即可。
2. rsync 同步：
   - 第一次多源 rsync 没用 `--relative`，曾把 `qwen3-8b-fixes.md`、`test_chunked_prefill.py` 临时同步到远程 repo 根目录；已删除这两个误同步副本。
   - 正确同步方式：`rsync -av -e 'ssh -S /tmp/ssh-sitian-10.232.195.203-new -o BatchMode=yes' --relative <files> sitian@10.232.195.203:/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/`。
3. GPU blocks matrix：
   - `gpu_blocks=1` 失败是当前实现边界：`required=2, gpu_blocks=1`，因为 blockwise prefill 需要 `prefix window blocks + 当前 chunk write blocks <= gpu staging slots`；不要把它当 correctness mismatch。
   - `gpu_blocks=2/4` 已通过；`gpu_blocks=4` 的 H2D/eviction 次数比 2 少，符合 staging slots 增多的预期。
4. Multi-prompt batch smoke：
   - 默认 `MAX_MODEL_LEN=4096`、`GPU_MEMORY_UTILIZATION=0.7`、`MULTI_PROMPT_COUNT=2` 初次失败在模型初始化阶段：`allocate_kv_cache()` 里 `assert auto_num_blocks > 0`，不是 attention correctness 失败。
   - 可用参数是降低 `MAX_MODEL_LEN=2048`、提高 `GPU_MEMORY_UTILIZATION=0.85`、缩短 `MULTI_PROMPT_REPEAT=24`，最终 `num_prompts=2` 通过。
5. 本地验证注意：
   - 本地 `python3 -m py_compile` 若写 macOS 系统 pyc cache 失败，可用 `PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache`。
   - 当前未跟踪的 `.agents/`、`.codex/`、`needle_sq_results/` 是无关本地/实验产物，之前提交均未纳入。

## 远程 S4 smoke 示例

```bash
ssh sitian@10.232.195.203 'cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge && CUDA_VISIBLE_DEVICES=7 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B \
  --max-output-len 64 \
  --temperature 0.0 \
  --ngram-size 5 \
  --max-draft-tokens 4 \
  --max-commit-events 0 \
  --mode candidate-only \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7 \
  --max-num-seqs 16 \
  --out-json profile_out/ngram_spec_s4_06b_candidate_n5d4_verify.json'
```
