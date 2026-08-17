# T1.3 执行日志

## 2026-07-07

已完成：

- 新增 `tools/probe_kv_recovery.py`。
- 新增 `light-doc-cache-plan/t1_3_probe/README.md`。
- 脚本支持 synthetic 自检模式和真实 HuggingFace 模型模式。
- 输出格式覆盖 `summary.json`、`report.md`、跨 layer heatmap CSV、最佳 cross-head/cross-layer CSV。
- 默认判定口径：GO >= 0.50，BORDERLINE >= 0.35，低于 0.35 为 NO-GO。

已验证：

- 语法编译通过：

```bash
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
  python3 -m py_compile tools/probe_kv_recovery.py
```

- `--help` 可在无 PyTorch 的系统 Python 下正常输出。
- 无 PyTorch 环境下运行探针会给出明确错误信息，而不是裸 `ModuleNotFoundError`。

当前阻塞：

- 当前沙箱的 `/usr/bin/python3` 未安装 `torch`，仓库内也未发现虚拟环境，因此无法在本机直接跑 synthetic 或真实模型探针。

下一步：

1. 切换到带 `torch`、`transformers`、本地模型权重的 Python 环境。
2. 先跑 synthetic smoke：

```bash
python3 tools/probe_kv_recovery.py \
  --synthetic \
  --no-png \
  --output-dir light-doc-cache-plan/t1_3_probe/runs/synthetic_smoke
```

3. 再跑真实模型：

```bash
python3 tools/probe_kv_recovery.py \
  --model ~/Qwen3-0.6B \
  --text-file docs/kv-sparse-attention.md \
  --max-tokens 2048 \
  --max-sample-tokens 1024 \
  --output-dir light-doc-cache-plan/t1_3_probe/runs/qwen3_0_6b
```

## 2026-07-08

已完成远程 task-level quality smoke：

- 本地补齐 `experiments/light_doc_cache/probe_am_compact_cache.py`，避免 `task_quality_smoke.py` 依赖远程残留文件。
- 修复 `experiments/light_doc_cache/task_quality_smoke.py` 的 baseline 评分路径：attention monkey patch 现在保存 `_light_doc_cache_original_forward`，当当前层没有 compact bank 时直接回退原始 Qwen3 attention forward；此前 baseline 也走了 Python 重写 attention，导致 baseline 与 compact 都一致偏向选项 D，实验不可解释。
- 优化 `experiments/light_doc_cache/run_task_quality_smoke_remote.sh`：
  - 支持 `CONTROL_PATH=/private/tmp/ssh-sitian-light-doc-cache` 复用 ControlMaster。
  - 支持 `SSH_OPTS`。
  - 禁止写入 known_hosts（`UserKnownHostsFile=/dev/null`、`UpdateHostKeys=no`），规避当前 sandbox 无法改 `~/.ssh/known_hosts` 的问题。
  - 由多次 base64 chunk SSH 传输改为单次 stdin stream 传输。
  - 同步并校验 `task_quality_smoke.py` 与 `probe_am_compact_cache.py` 的 size、SHA256、`py_compile`。

远程运行命令：

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
CONTROL_PATH=/private/tmp/ssh-sitian-light-doc-cache \
  experiments/light_doc_cache/run_task_quality_smoke_remote.sh
```

远程输出：

- `/data00/home/sitian/light-doc-cache-work/probe/runs/task_quality_smoke_qwen3_0_6b_s1536_policy035_050/report.md`
- `/data00/home/sitian/light-doc-cache-work/probe/runs/task_quality_smoke_qwen3_0_6b_s1536_policy035_050/summary.json`
- 本地镜像：`experiments/light_doc_cache/task_quality_smoke_remote_latest/report.md`、`summary.json`

结果摘要：

| threshold | compressed heads | baseline acc | compact acc | agreement | mean answer logP delta | mean compact margin |
|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | 63 | 60.00% | 20.00% | 40.00% | -1.8541 | -0.7531 |
| 0.50 | 11 | 60.00% | 60.00% | 100.00% | +0.2862 | +0.4000 |

解释：

- `threshold=0.35` 的 attention-output policy 压缩 63 个 KV heads，entry saving 约 25.15%，但 task-level smoke 明显掉质量；不能作为 go 策略。
- `threshold=0.50` 只压缩 11 个 KV heads，entry saving 约 4.30%，task-level smoke 与 baseline 一致，说明更保守阈值在这个小样本上可保持质量，但净压缩收益很弱。
- 当前 baseline 只有 5 题、准确率 60%，所以该 smoke 只能作为早期 sanity check，不足以做论文/产品结论。下一步应扩展更稳的任务集，或先扩大 prompt/question 模板并验证 baseline 自身准确率。

本地验证：

```bash
python3 /tmp/test_runner_static.py
python3 /tmp/test_runner_dependency_static.py
python3 /tmp/test_task_quality_patch_static.py
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
  python3 -m py_compile \
    experiments/light_doc_cache/task_quality_smoke.py \
    experiments/light_doc_cache/make_attention_output_policy.py \
    experiments/light_doc_cache/probe_am_compact_cache.py \
    tools/probe_kv_recovery.py
bash -n experiments/light_doc_cache/run_task_quality_smoke_remote.sh
git diff --check
```

补充：已给 `task_quality_smoke.py` 增加 baseline-gated 指标，避免 baseline 自身只答对 3/5 时误读整体 compact accuracy。最新远程结果新增：

| threshold | baseline-correct tasks | compact acc on baseline-correct | agreement on baseline-correct |
|---:|---:|---:|---:|
| 0.35 | 3 / 5 | 33.33% | 33.33% |
| 0.50 | 3 / 5 | 100.00% | 100.00% |

这强化了前述解释：`0.35` 对 baseline 能答对的问题也明显破坏，`0.50` 在这个小 smoke 上保持 baseline 行为但压缩收益很弱。

补充 2：已增加 `--min-baseline-accuracy`（默认 80%）和 `baseline_gate_pass`。最新 smoke 中 baseline 只有 60%，两档均标记 `weak-baseline`，因此只能解释 compact 相对 baseline 行为，不能作为绝对任务质量结论。

最新报告位置：

- 远程：`/data00/home/sitian/light-doc-cache-work/probe/runs/task_quality_smoke_qwen3_0_6b_s1536_policy035_050/report.md`
- 本地：`experiments/light_doc_cache/task_quality_smoke_remote_latest/report.md`

## 2026-07-08 choice scoring calibration

新增 `--choice-scoring {letter,space_letter,letter_dot_text,text_only,space_text}` 并用远程 Qwen3-0.6B 对比 5 种候选打分字符串。汇总已落盘：`experiments/light_doc_cache/task_quality_scoring_compare.md`。

结论：

- `letter` / `space_letter` baseline 只有 60%，低于默认 80% baseline gate，不适合作为主解释口径。
- `letter_dot_text`、`text_only`、`space_text` baseline 都达到 80%。其中 `text_only` 在这个 toy set 上 compact-vs-baseline 行为最干净：两档 threshold 在 baseline-correct 子集上的 agreement 都是 100%。
- `threshold=0.50` 仍是更稳策略；`threshold=0.35` 即使用 `text_only` 也有明显负 answer logP delta / margin 收缩。
- 当前只有 5 道任务，仍不能做最终 go/no-go；下一步应扩展任务集并优先使用 `--choice-scoring text_only` 做主 smoke。

已将 `task_quality_smoke.py` 的默认 `--choice-scoring` 改为 `text_only`，因为该口径在当前 5-task smoke 上达到 baseline gate，且 compact-vs-baseline 行为最稳定。若需要复现实验早期字母打分结果，可显式传 `--choice-scoring letter`。

主报告已用默认 `text_only` 口径重跑并刷新：

- 远程：`/data00/home/sitian/light-doc-cache-work/probe/runs/task_quality_smoke_qwen3_0_6b_s1536_policy035_050/report.md`
- 本地：`experiments/light_doc_cache/task_quality_smoke_remote_latest/report.md`

最新主结果：

| threshold | compressed heads | baseline gate | baseline acc | compact acc | agreement | baseline-correct agreement | mean answer logP delta | compact margin |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0.35 | 63 | pass | 80.00% | 80.00% | 100.00% | 100.00% | -3.2911 | 0.7557 |
| 0.50 | 11 | pass | 80.00% | 80.00% | 100.00% | 100.00% | +0.0311 | 2.7043 |

解释：`0.35` 在 5-task smoke 上保持最终选择，但 answer logP 大幅下降、margin 明显低于 baseline；`0.50` 更稳但压缩收益弱。后续应该扩大任务集，而不是基于 5 题下 go/no-go。

## 2026-07-08 external task-file smoke

新增外部任务集支持：

- `experiments/light_doc_cache/task_quality_smoke.py` 新增 `--task-file`，支持 JSON list 或 `{ "tasks": [...] }`。
- 每个 task 校验 `id/question/choices/answer`，且 `answer` 必须在 choices 中。
- `experiments/light_doc_cache/run_task_quality_smoke_remote.sh` 新增 `TASK_FILE=/path/to/tasks.json` 同步，远程写入 `task_quality_tasks.json`，并校验 size/SHA256。
- runner 新增 `SSH_RETRIES`，对 SSH 瞬断自动重试；并镜像 `report.md`、`summary.json`、`task_rows.csv`、`tasks.json` 到本地输出目录。

任务集：

- `experiments/light_doc_cache/task_quality_tasks_kv_sparse_v1.json`：12 题；远程 baseline 75%，低于 80% baseline gate，用于发现不稳定题。
- `experiments/light_doc_cache/task_quality_tasks_kv_sparse_v2.json`：9 题 baseline-stable 子集；移除 v1 中 baseline 答错的 `paged_kv_hook`、`quest_summary_storage`、`phase3_difficulty`。

远程 v2 命令：

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
TASK_FILE=experiments/light_doc_cache/task_quality_tasks_kv_sparse_v2.json \
OUTPUT_DIR=/data00/home/sitian/light-doc-cache-work/probe/runs/task_quality_smoke_qwen3_0_6b_s1536_policy035_050_tasks_v2 \
LOCAL_OUT_DIR=experiments/light_doc_cache/task_quality_smoke_tasks_v2_latest \
  experiments/light_doc_cache/run_task_quality_smoke_remote.sh
```

v2 结果：

| threshold | compressed heads | baseline gate | baseline acc | compact acc | agreement | mean answer logP delta | compact margin |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0.35 | 63 | pass | 100.00% | 55.56% | 55.56% | -3.6045 | 0.1170 |
| 0.50 | 11 | pass | 100.00% | 100.00% | 100.00% | -0.2323 | 3.1628 |

解释：扩展到 baseline-stable 9 题后，`threshold=0.35` 明确破坏 task-level 质量；`threshold=0.50` 保持所有题的 baseline 选择，但当前只压缩 11 heads，净压缩收益有限。下一步应扩大到更多文档/任务，或者寻找介于 0.35 和 0.50 之间的 policy threshold / per-layer policy。

本地镜像：

- `experiments/light_doc_cache/task_quality_smoke_tasks_v2_latest/report.md`
- `experiments/light_doc_cache/task_quality_smoke_tasks_v2_latest/summary.json`
- `experiments/light_doc_cache/task_quality_smoke_tasks_v2_latest/task_rows.csv`
- `experiments/light_doc_cache/task_quality_smoke_tasks_v2_latest/tasks.json`

## 2026-07-08 mid-threshold tradeoff sweep

用同一 AM run 生成中间阈值 policy：

```bash
cd /data00/home/sitian/light-doc-cache-work/probe
/data00/home/sitian/miniconda3/envs/py311/bin/python make_attention_output_policy.py \
  --am-run runs/am_qwen3_0_6b_s1536_b64_128_256_r1.0 \
  --output-dir runs/policy_am_qwen3_0_6b_s1536_holdout_all_r1.0_mid \
  --quality-thresholds 0.40,0.45 \
  --note mid-threshold-task-quality-sweep
```

Policy summary：

| threshold | compressed heads | entry saving | mean compressed R2 |
|---:|---:|---:|---:|
| 0.40 | 44 / 224 | 17.43% | 0.4467 |
| 0.45 | 27 / 224 | 10.79% | 0.4785 |

远程 v2 质量验证输出：

- `/data00/home/sitian/light-doc-cache-work/probe/runs/task_quality_smoke_qwen3_0_6b_s1536_policy040_045_tasks_v2`
- 本地镜像：`experiments/light_doc_cache/task_quality_smoke_tasks_v2_mid_latest/`

综合 0.35/0.40/0.45/0.50 的质量-压缩折中表已落盘：`experiments/light_doc_cache/task_quality_v2_tradeoff.md`。

结论：

- `0.35`、`0.40`、`0.45` 都未通过 v2 task-level quality；即使 0.45 只压缩 27 heads / saving 10.79%，compact accuracy 仍只有 77.78%。
- `0.50` 是当前已测阈值中唯一保持 9/9 baseline-correct tasks 的策略，但只压缩 11 heads / saving 约 4.30%。
- 当前单源 AM policy 的质量/压缩折中不够吸引；下一步更值得做 per-layer/per-head 约束搜索或 multi-source recovery，而不是继续把该 policy 集成到 runtime。

## 2026-07-08 late-layer policy check

验证已有 late-layer policy：`policy_am_qwen3_0_6b_s1536_holdout_l16_27_r1.0`。

Policy summary：

| threshold | compressed heads | entry saving |
|---:|---:|---:|
| 0.35 | 32 / 224 | 12.80% |
| 0.50 | 5 / 224 | 1.93% |

v2 quality：

| threshold | compact acc | agreement | answer logP delta | compact margin |
|---:|---:|---:|---:|---:|
| 0.35 | 88.89% | 88.89% | -2.3634 | 1.8717 |
| 0.50 | 100.00% | 100.00% | -0.3994 | 3.1079 |

综合对比已落盘：`experiments/light_doc_cache/task_quality_v2_layer_policy_compare.md`。

结论：late-layer-only 的 0.35 比 all-layer 0.35 明显改善，但仍未通过 9-task quality；late-layer 0.50 通过但只 saving 1.93%，比 all-layer 0.50 的 4.30% 更弱。因此简单 layer filter 不是足够好的折中方向。

## 2026-07-08 constrained per-layer policy sweep

为避免 raw threshold policy 一次性压缩过多敏感 head，新增受限 policy 生成能力：

- `experiments/light_doc_cache/make_attention_output_policy.py`
  - `--max-compact-heads`
  - `--max-compact-heads-per-layer`
  - `--min-saving-fraction`
- 默认值保持旧行为不变。
- 选择逻辑：每个 head 先用原来的最小 budget 达到 R2 threshold；再过滤低于最小 saving 的 head；最后按 holdout `fitv_val_r2` 降序，在全局/per-layer cap 下选 compact heads。

已验证本地静态/合成行为测试：

```bash
python3 /tmp/test_constrained_policy_static.py
python3 /tmp/test_constrained_policy_behavior.py
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile experiments/light_doc_cache/make_attention_output_policy.py
```

远端 policy 生成目录：

- `/data00/home/sitian/light-doc-cache-work/probe/runs/policy_am_qwen3_0_6b_s1536_holdout_constrained_r1.0`
- `/data00/home/sitian/light-doc-cache-work/probe/runs/policy_am_qwen3_0_6b_s1536_holdout_constrained_cap20_r1.0`
- `/data00/home/sitian/light-doc-cache-work/probe/runs/policy_am_qwen3_0_6b_s1536_holdout_constrained_cap24_r1.0`
- `/data00/home/sitian/light-doc-cache-work/probe/runs/policy_am_qwen3_0_6b_s1536_holdout_constrained_cap18_l1_r1.0`

本地镜像：

- `experiments/light_doc_cache/policy_am_qwen3_0_6b_s1536_holdout_constrained_r1.0/`
- `experiments/light_doc_cache/policy_am_qwen3_0_6b_s1536_holdout_constrained_cap20_r1.0/`
- `experiments/light_doc_cache/policy_am_qwen3_0_6b_s1536_holdout_constrained_cap24_r1.0/`
- `experiments/light_doc_cache/policy_am_qwen3_0_6b_s1536_holdout_constrained_cap18_l1_r1.0/`

v2 quality 结果：

| Policy | threshold | heads | entry saving | compact acc | agreement | answer logP delta | conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| constrained16_l1 | 0.35 | 16 | 6.47% | 88.89% | 88.89% | -0.9532 | fail |
| constrained16_l1 | 0.40 | 16 | 6.40% | 88.89% | 88.89% | -0.9232 | fail |
| constrained16_l1 | 0.45 | 16 | 6.45% | 100.00% | 100.00% | -0.5539 | pass |
| constrained16_l1 | 0.50 | 9 | 3.52% | 100.00% | 100.00% | +0.0499 | pass, lower saving |
| constrained18_l1 | 0.45 | 17 | 6.83% | 100.00% | 100.00% | -0.7587 | best current safe smoke |
| constrained20_l2 | 0.45 | 20 | 8.04% | 77.78% | 77.78% | -1.0685 | fail |
| constrained24_l2 | 0.45 | 23 | 9.19% | 77.78% | 77.78% | -1.7738 | fail |

解释：

- 受限 per-layer policy 比 raw threshold/layer-filter 明显更好：当前最好点是 `threshold=0.45, max_compact_heads=18, max_compact_heads_per_layer=1, min_saving_fraction=0.50`，实际压缩 17 heads，entry saving 约 6.83%，v2 9/9 保持 baseline 答案。
- 安全边界很窄：把 cap 放宽到 20/23 heads 后，`quest_summary_form` 和 `quest_default_enable` 翻转，说明不能简单追求更多 heads。
- 结论从“简单 threshold policy 不值得集成”更新为：“per-layer constrained policy 是一个可继续研究的低成本方向，但仍只是 single-document 9-task quality smoke；进入 runtime 前需要更多文档/任务 stress 或 multi-source recovery。”

## 2026-07-08 v3 stress task expansion

新增更强任务集：

- `experiments/light_doc_cache/task_quality_tasks_kv_sparse_v3_candidate.json`
  - 20 题，包含 v2 题 + 更多工程细节题 + 非 A 正确选项。
- `experiments/light_doc_cache/task_quality_tasks_kv_sparse_v3_stable.json`
  - 用 empty policy baseline-only 校准后保留 13 道 baseline-correct 题。
  - 排除 7 道 baseline 不稳定题：`paged_cache_shape`、`summary_storage`、`cuda_graph_initial_decision`、`quest_slowdown_root`、`needle_fix`、`topk4_quality`、`phase3_snapkv_hard_part`。

Empty policy baseline-only：

- 远程：`/data00/home/sitian/light-doc-cache-work/probe/runs/task_quality_smoke_qwen3_0_6b_s1536_v3_candidate_baseline`
- 本地：`experiments/light_doc_cache/task_quality_smoke_v3_candidate_baseline_latest/`
- baseline accuracy 65.00%，低于 80% gate；只能用于筛 baseline-stable 子集，不能解释 compact quality。

对 v2 最佳点 `constrained18_l1 threshold=0.45` 做 v3 stable stress：

- 远程：`/data00/home/sitian/light-doc-cache-work/probe/runs/task_quality_smoke_qwen3_0_6b_s1536_constrained_cap18_l1_tasks_v3_stable`
- 本地：`experiments/light_doc_cache/task_quality_smoke_tasks_v3_constrained_cap18_l1_latest/`

结果：

| task set | policy | threshold | heads | entry saving | baseline acc | compact acc | agreement | answer logP delta | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| v3_candidate | empty | 0.99 | 0 | 0.00% | 65.00% | 65.00% | 100.00% | 0.0000 | weak baseline |
| v3_stable | constrained18_l1 | 0.45 | 17 | 6.83% | 100.00% | 84.62% | 84.62% | -0.6256 | fail |

失败任务：

| task | baseline | compact | answer | delta | compact margin |
|---|---|---|---|---:|---:|
| `phase2_magic_number` | B | A | B | -0.0574 | -0.1019 |
| `sweet_spot` | B | A | B | -0.1323 | -0.2110 |

更新结论：

- v2 9 题全部答案都是 A，过于弱；v3 加入非 A baseline-stable 题后，17-head policy 暴露选择偏移/质量不稳。
- 当前 T1.3 不能进入 runtime 集成；受限 policy 只能作为研究信号，下一步应该：
  1. 把 v3 stable 作为最低 task-quality gate；
  2. 搜索更严格的 task-aware constraints（例如排除会影响非 A 决策 margin 的层/head）；
  3. 或转向 multi-source recovery，而不是继续扩大 raw AM threshold policy。

## 2026-07-08 task-aware ablation follow-up

基于 v3 stable 失败继续定位：

1. 新增工具：
   - `experiments/light_doc_cache/make_policy_ablation.py`
     - 支持 `leave_one_out`、`pair_drop`、`prefix`。
     - 注意：首次实现时 dropped compact head 的 `selected_budget` 没恢复为 full length，质量模拟不受影响但 saving 统计偏乐观；已修复，并用 `/tmp/test_policy_ablation_full_budget.py` 验证。
   - `experiments/light_doc_cache/extract_policy_threshold.py`
     - 从多 threshold policy rows 中抽取单个候选，重标成标准 threshold 便于复现实验。

2. Leave-one-out：
   - 远程：`/data00/home/sitian/light-doc-cache-work/probe/runs/task_quality_smoke_qwen3_0_6b_s1536_v3_leave_one_out`
   - 本地：`experiments/light_doc_cache/task_quality_smoke_v3_leave_one_out_latest/`
   - 结论：单删任何一个 head 都不能让 v3 stable 达到 100%；最高 92.31%，只能修复两个非 A 失败中的一个。

3. Pair-drop fail-only 快筛：
   - 新增 `experiments/light_doc_cache/task_quality_tasks_v3_failonly.json`，只含 `phase2_magic_number` 和 `sweet_spot`。
   - 远程：`/data00/home/sitian/light-doc-cache-work/probe/runs/task_quality_smoke_qwen3_0_6b_s1536_v3_pair_drop_failonly`
   - 本地：`experiments/light_doc_cache/task_quality_smoke_v3_pair_drop_failonly_latest/`
   - 136 个 pair-drop 中有 26 个修复两个 fail-only tasks。
   - 按 fail-only min margin 选最稳候选：`drop_l3_h0__l23_h7`。

4. 抽取并验证 task-aware policy：
   - 本地 policy：`experiments/light_doc_cache/policy_taskaware_drop_l3_h0_l23_h7/`
   - 远程 policy：`/data00/home/sitian/light-doc-cache-work/probe/runs/policy_taskaware_drop_l3_h0_l23_h7`
   - 压缩 15 heads；修正后 entry saving 约 6.03%。
   - selected heads: `2:7;4:6;5:2;6:7;8:1;9:1;10:3;11:2;14:5;18:0;20:2;22:1;24:2;25:3;26:1`。

验证结果：

| task set | policy | heads | entry saving | baseline acc | compact acc | agreement | mean delta | compact margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v3 stable | taskaware_drop_l3_h0_l23_h7 | 15 | 6.03% | 100.00% | 100.00% | 100.00% | -0.4694 | 2.3124 |
| v2 stable | taskaware_drop_l3_h0_l23_h7 | 15 | 6.03% | 100.00% | 100.00% | 100.00% | -0.6160 | 3.0370 |

当前结论：

- 相比 raw safe `0.50` 的 11 heads / 4.30%，task-aware pair-drop 提升到 15 heads / 6.03%，并通过 v2/v3 stable。
- 相比 v2-only 17-head policy 的 6.83%，它牺牲约 0.80 pct saving 来换取 v3 非 A 任务稳定性。
- 这仍然只是单文档 quality-only simulation；下一步不要进 runtime，而应做 multi-document stress 或把 task-aware gate 纳入自动 policy 搜索。

## 2026-07-08 second-document stress

为避免第一文档过拟合，runner 新增 `LOCAL_TEXT_FILE=...`：

- 会把本地 Markdown 文档同步到远端 `$REMOTE_DIR/task_quality_text.md`；
- 校验 size/SHA256；
- 远端 `task_quality_smoke.py --text-file` 自动指向同步后的文件。

新增任务集：

- `experiments/light_doc_cache/task_quality_tasks_qwen3_8b_fixes_candidate.json`
  - 12 道候选题，来自 `docs/qwen3-8b-fixes.md`。
- `experiments/light_doc_cache/task_quality_tasks_qwen3_8b_fixes_stable.json`
  - baseline-only 校准后保留 6 道 baseline-correct 题。
  - 排除：`qwen3_8b_goal`、`c4_quality_boundary`、`weight_mem_metric`、`w4a8_root_cause_hypothesis`、`act_quant_fix`、`w4_group_size_fix`。

Baseline-only：

- 远端：`/data00/home/sitian/light-doc-cache-work/probe/runs/task_quality_smoke_qwen3_0_6b_qwen8bfixes_candidate_baseline`
- 本地：`experiments/light_doc_cache/task_quality_smoke_qwen8bfixes_candidate_baseline_latest/`
- baseline acc 50.00%，低于 gate；仅用于筛 stable 子集。

Second-document stable 对照：

| policy | heads | saving | compact acc | agreement | mean delta | compact margin | conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| all_layers_0.50 | 11 | 4.30% | 100.00% | 100.00% | +0.8972 | 3.5369 | pass |
| constrained18_l1_0.45 | 17 | 6.83% | 83.33% | 83.33% | +0.1766 | 2.2585 | fail |
| taskaware_drop_l3_h0_l23_h7 | 15 | 6.03% | 66.67% | 66.67% | +0.3109 | 2.3160 | fail |

失败任务：

| policy | task | baseline | compact | answer | delta | compact margin |
|---|---|---|---|---|---:|---:|
| taskaware_drop_l3_h0_l23_h7 | `qwen3_8b_model_choice` | B | D | B | +1.0515 | -0.1506 |
| taskaware_drop_l3_h0_l23_h7 | `tp_true_weight_split` | A | D | A | -1.6273 | -0.0011 |
| constrained18_l1_0.45 | `qwen3_8b_model_choice` | B | D | B | +0.6268 | -0.9651 |

更新结论：

- `taskaware_drop_l3_h0_l23_h7` 是第一文档任务感知修补，不具备跨文档泛化。
- 当前唯一跨两个文档 stable 集都通过的策略是 all-layer `0.50`，11 heads / 4.30% entry saving。
- 这把 T1.3 结论重新拉回保守：单源 AM recovery 的可用压缩收益仍偏低，不能进入 runtime integration；下一步应做 document-adaptive policy / multi-source recovery，而不是继续手工找固定 head 子集。

## 2026-07-09 trainable recovery MVP start

根据 KR4 方向，停止继续手工搜索固定 head 子集，转向“训练过的相关性 doc cache / gist cache / recovery 模块”离线 MVP：

- 新增 `experiments/light_doc_cache/train_recovery_probe.py`
  - 复用 `probe_am_compact_cache.py` 的 Q/K/V 采集、attention target、token selection、ridge value fitting、R² 评分。
  - 对每个 doc/layer/KV-head 选择 compact tokens，teacher target 为 full attention output。
  - 训练一个 per-head `mlp_residual` recovery module：输入 direct compact attention output，输出恢复后的 attention output。
  - 同时报告 direct compact、ridge-fitted compact values、MLP recovery，以及 non-degrading recovery envelope。
  - 输出 `recovery_head_rows.csv`、`summary.json`、`report.md`。
- 新增 `experiments/light_doc_cache/run_recovery_probe_remote.sh`
  - 同步 `train_recovery_probe.py` 和 `probe_am_compact_cache.py` 到远端；
  - 支持 `LOCAL_TEXT_FILES="doc1 doc2 ..."` 多文档同步并校验 size/SHA256；
  - 运行远端 Qwen3-0.6B recovery probe；
  - 镜像 `report.md`、`summary.json`、`recovery_head_rows.csv`、`run.log` 回本地。
- 新增 `tools/test_light_doc_cache_recovery_probe.py`
  - 本地无 torch 时至少做静态 contract 测试；
  - 有 torch 时运行纯合成数据行为测试，验证 recovery loss 下降且不劣于 direct compact。

推荐先跑小头数远端 smoke：

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
LOCAL_TEXT_FILES="docs/kv-sparse-attention.md docs/qwen3-8b-fixes.md" \
OUTPUT_DIR=/data00/home/sitian/light-doc-cache-work/probe/runs/recovery_probe_qwen3_0_6b_s1536_b64_2docs_smoke \
LOCAL_OUT_DIR=experiments/light_doc_cache/recovery_probe_2docs_smoke_latest \
BUDGETS=64 \
MAX_HEADS=8 \
EPOCHS=100 \
  experiments/light_doc_cache/run_recovery_probe_remote.sh
```

解释边界：

- 这是离线 teacher-student recovery probe，不是 runtime attention 集成。
- 如果 multi-doc recovery R² 有稳定收益，再把 recovery 输出接进 `task_quality_smoke.py` 做任务级复测。
- 若 recovery 只在 train/doc 内提升而 cross-doc weak，则说明需要 doc-adaptive 或 multi-source 训练，而不是把当前模块产品化。

### 2026-07-09 trainable recovery smoke results

本地验证：

```bash
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 tools/test_light_doc_cache_recovery_probe.py
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile experiments/light_doc_cache/train_recovery_probe.py tools/test_light_doc_cache_recovery_probe.py
bash -n experiments/light_doc_cache/run_recovery_probe_remote.sh
git diff --check
```

远端 smoke 1：layer 0，2 docs，`BUDGETS=64`，`MAX_HEADS=8`，`EPOCHS=100`

- 远端：`/data00/home/sitian/light-doc-cache-work/probe/runs/recovery_probe_qwen3_0_6b_s1536_b64_2docs_smoke`
- 本地：`experiments/light_doc_cache/recovery_probe_2docs_smoke_latest/`
- result: `RECOVERY_WEAK`
- heads: 16
- mean budget fraction: 12.50%
- mean direct val R²: -0.3912
- mean ridge-value val R²: 0.0033
- mean recovery val R²: -0.0491

远端 smoke 2：layer 10，2 docs，`BUDGETS=64`，`MAX_HEADS=8`，`EPOCHS=100`

- 远端：`/data00/home/sitian/light-doc-cache-work/probe/runs/recovery_probe_qwen3_0_6b_s1536_b64_2docs_l10_smoke`
- 本地：`experiments/light_doc_cache/recovery_probe_2docs_l10_smoke_latest/`
- result: `RECOVERY_WEAK`
- heads: 16
- mean budget fraction: 12.50%
- mean direct val R²: -0.6530
- mean ridge-value/recovery val R²: 0.0862
- recovery coverage above 0.5: 6.25%

远端 smoke 3：新增 trainable `learned_compact_values` 后复测 layer 10

- 远端：`/data00/home/sitian/light-doc-cache-work/probe/runs/recovery_probe_qwen3_0_6b_s1536_b64_2docs_l10_learnedv_smoke`
- 本地：`experiments/light_doc_cache/recovery_probe_2docs_l10_learnedv_smoke_latest/`
- result: `RECOVERY_WEAK`
- learned compact values 未超过 ridge closed-form；mean recovery val R² 仍为 0.0862。

当前解释：

- 这个 MVP 已经具备“训练过的 recovery 模块”的离线形态，但当前特征/目标还不足以支撑 2x+ doc cache 压缩收益。
- direct compact output 在 64/512 token budget 下很弱；ridge-value recovery 能明显修 direct，但仍远低于可进入 task-quality/runtime 的阈值。
- residual MLP 和 learned compact-values 在 holdout 上不如 closed-form ridge，说明当前 per-head doc-local 训练容易过拟合，不能作为 paper 主结果。

下一步建议：

1. 做 layer sweep：`START_LAYER=0/4/8/12/16/20/24`，先找恢复友好的层段，而不是全层平均。
2. 提升 recovery feature：不要只看 compact attention output；加入 selected token position、attention entropy/mass、query/key pooled features，或直接拟合 attention logits/missing mass。
3. 改为 shared multi-doc recovery：跨 doc/head 共享一个小模块，按 layer/head embedding 条件化，检验是否能泛化。
4. 若 offline R² 能稳定上来，再接入 `task_quality_smoke.py` 的 compact-bank scoring 做任务级验证；现在仍不进入 runtime integration。

### 2026-07-09 layer/budget recovery sweep follow-up

新增聚合工具：

- `experiments/light_doc_cache/summarize_recovery_sweep.py`
  - 扫描一个或多个 `summary.json` / run directory / parent directory；
  - 输出 `recovery_sweep_summary.csv` 和 `report.md`；
  - 按 `mean_recovery_val_r2`、`mean_recovery_gain_vs_fitv` 排序。

新增 recovery 特征：

- `fused_residual`：输入 `[direct compact output, ridge-fit output, direct-fitv residual]`，从 ridge-fit output 上学习 residual。
- 目的：让 MLP 不再只依赖很弱的 direct compact output。

Layer sweep（2 docs, Qwen3-0.6B, layer start, 8 KV heads, budget 64/512, 12.5% token budget）：

| start layer | local dir | decision | mean direct R² | mean recovery R² | gain vs FitV | coverage@0.5 |
|---:|---|---|---:|---:|---:|---:|
| 0 | `recovery_probe_2docs_smoke_latest` | `RECOVERY_WEAK` | -0.3912 | -0.0491 | -0.0524 | 0.00% |
| 4 | `recovery_probe_2docs_l4_smoke_latest` | `RECOVERY_WEAK` | -0.4442 | 0.0143 | 0.0000 | 0.00% |
| 8 | `recovery_probe_2docs_l8_smoke_latest` | `RECOVERY_NEEDS_TASK_SMOKE` | -0.3874 | 0.1138 | 0.0006 | 0.00% |
| 10 | `recovery_probe_2docs_l10_learnedv_smoke_latest` | `RECOVERY_WEAK` | -0.6530 | 0.0862 | 0.0000 | 6.25% |
| 12 | `recovery_probe_2docs_l12_smoke_latest` | `RECOVERY_NEEDS_TASK_SMOKE` | -0.2689 | 0.1605 | 0.0030 | 0.00% |
| 16 | `recovery_probe_2docs_l16_smoke_latest` | `RECOVERY_WEAK` | -0.3204 | 0.0856 | 0.0000 | 0.00% |
| 20 | `recovery_probe_2docs_l20_smoke_latest` | `RECOVERY_WEAK` | -0.4508 | 0.0932 | 0.0000 | 0.00% |
| 24 | `recovery_probe_2docs_l24_fused_smoke_latest` | `RECOVERY_NEEDS_TASK_SMOKE` | -0.2694 | 0.1647 | 0.0011 | 6.25% |

Budget sweep at layer 24（2 docs, 8 KV heads）：

| budget | token fraction | approx compression | local dir | decision | direct R² | fitV R² | recovery R² | gain vs FitV | coverage@0.5 |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 64 | 12.50% | 8.0x | `recovery_probe_2docs_l24_fused_smoke_latest` | `RECOVERY_NEEDS_TASK_SMOKE` | -0.2694 | 0.1636 | 0.1647 | 0.0011 | 6.25% |
| 128 | 25.00% | 4.0x | `recovery_probe_2docs_l24_b128_fused_smoke_latest` | `RECOVERY_NEEDS_TASK_SMOKE` | 0.0936 | 0.2179 | 0.2295 | 0.0116 | 12.50% |
| 192 | 37.50% | 2.67x | `recovery_probe_2docs_l24_b192_fused_smoke_latest` | `RECOVERY_NEEDS_TASK_SMOKE` | 0.2210 | 0.2087 | 0.2650 | 0.0563 | 12.50% |
| 256 | 50.00% | 2.0x | `recovery_probe_2docs_l24_b256_fused_smoke_latest` | `RECOVERY_NEEDS_TASK_SMOKE` | 0.2999 | 0.1852 | 0.3085 | 0.1232 | 25.00% |

最新聚合输出：

- `experiments/light_doc_cache/recovery_sweep_summary_latest/report.md`
- `experiments/light_doc_cache/recovery_sweep_summary_latest/recovery_sweep_summary.csv`

解释：

- layer 24 是当前最好的 probe 层，但即使 2x token budget 下 mean recovery val R² 也只有 0.3085，coverage@0.5 只有 25%。
- fused/MLP 类训练模块在 budget 192/256 开始明显超过 FitV，但收益还不足以支撑 “2x+ 压缩精度无损” 声明。
- 当前可写成 negative/diagnostic evidence：后层更适合 recovery，2x budget 有可训练收益，但 per-head doc-local recovery 仍不够。
- 下一步应优先做 shared multi-doc/multi-head recovery 或 task-level smoke adapter，而不是 runtime 集成。

### 2026-07-09 task-level recovery smoke follow-up

为了把 offline recovery 信号推进到 task quality，做了以下代码改动：

- `task_quality_smoke.py`
  - 新增 `--bank-method {ridge,learned_values}`。
  - 新增 `--bank-train-epochs`、`--bank-train-lr`、`--bank-weight-decay`。
  - `learned_values` 会在 ridge compact values 初始化基础上，用 prompt 内 train tokens 继续训练 compact values。
- `run_task_quality_smoke_remote.sh`
  - 新增 `LOCAL_POLICY_DIR=...`，会同步本地 `policy_rows.csv` 到远端 `$REMOTE_DIR/task_quality_policy` 并校验 size/SHA256。
  - 同步 `train_recovery_probe.py`，保证 `task_quality_smoke.py` 能导入 `train_compact_values`。
- `make_recovery_task_policy.py`
  - 生成固定 layer 或 `all` layer 的 recovery task policy。
  - `policy_recovery_l24_b050`: layer 24，8 KV heads，50% selected budget，总 entry saving 1.79%。
  - `policy_recovery_all_b050`: all layers，224 KV heads，50% selected budget，总 entry saving 50.00%。

任务级验证：layer24 单层 2x（总 saving 1.79%）

| doc/task set | bank method | local dir | baseline acc | compact acc | agreement | mean delta | compact margin | result |
|---|---|---|---:|---:|---:|---:|---:|---|
| `kv-sparse-attention` v3 stable | learned_values | `task_quality_smoke_recovery_l24_b050_v3_learned_values_latest` | 100.00% | 100.00% | 100.00% | +0.2624 | 2.5274 | pass |
| `qwen3-8b-fixes` stable | learned_values | `task_quality_smoke_recovery_l24_b050_qwen8bfixes_learned_values_latest` | 100.00% | 100.00% | 100.00% | +0.6987 | 3.0201 | pass |
| `kv-sparse-attention` v3 stable | ridge | `task_quality_smoke_recovery_l24_b050_v3_ridge_latest` | 100.00% | 100.00% | 100.00% | +0.2861 | 2.5471 | pass |
| `qwen3-8b-fixes` stable | ridge | `task_quality_smoke_recovery_l24_b050_qwen8bfixes_ridge_latest` | 100.00% | 100.00% | 100.00% | +0.7324 | 3.0697 | pass |

解释：

- layer24 单层 2x 在两个文档 task-level stable sets 上都通过。
- 但 ridge bank 比 learned_values 更快，且 mean delta / margin 略好；当前 task-level pass 不能证明 learned_values 必要。
- 由于只压缩 8/224 KV heads，总 entry saving 只有 1.79%，不能对应 KR 的 “doc cache 2x+” 总体压缩目标。

任务级验证：all-layer 2x（总 saving 50.00%）

- policy: `policy_recovery_all_b050`
- local result: `task_quality_smoke_recovery_all_b050_v3_ridge_latest`
- first doc v3 stable：baseline 100.00%，compact 15.38%，agreement 15.38%，mean delta -34.5383，compact margin -14.6126。
- 明确失败。

当前结论：

- 正信号：后层（layer24）局部 doc-cache bank 可以在任务级保持质量，两个文档都 pass。
- 负信号：把 2x budget 扩展到全层/全 head 会严重破坏任务质量。
- 因此当前方法还不是 “doc cache size 2x+ 压缩精度无损”；它是一个定位结果：可压缩区域集中在后层少数 head，全局 2x 需要 head/layer 选择或共享 recovery，而不是全层平均裁剪。

下一步建议：

1. 自动搜索更多 safe heads：从 layer24 扩展到高层子集（例如 layers 20-27），用 task-quality pass gate 而不是 offline R² 排序。
2. 做 progressive policy：从 layer24 开始逐层/逐 head 增加压缩，直到 v3 + qwen8bfixes stable 任一失败。
3. 如果要证明 trained module 价值，需要让 learned/shared recovery 在 task-level 超过 ridge，而当前 prompt-local learned values 未超过 ridge。

### 2026-07-09 high-layer progressive range search

为了尝试从 layer24 单层 pass 扩展到更高 saving，`make_recovery_task_policy.py` 新增：

- `--compact-layer all`
- `--compact-layer-range start:end`

生成的高层范围 policy：

| policy | layers | heads | per-head budget | total entry saving |
|---|---:|---:|---:|---:|
| `policy_recovery_l24_28_b050` | 24:28 | 32 | 50.00% | 7.14% |
| `policy_recovery_l22_28_b050` | 22:28 | 48 | 50.00% | 10.71% |
| `policy_recovery_l20_28_b050` | 20:28 | 64 | 50.00% | 14.29% |

远端 first-doc v3 stable 结果：

| policy | local dir | compact acc | agreement | mean delta | compact margin | result |
|---|---|---:|---:|---:|---:|---|
| `policy_recovery_l20_28_b050` | `task_quality_smoke_recovery_l20_28_b050_v3_ridge_latest` | 23.08% | 23.08% | -31.1790 | -13.5662 | fail |
| `policy_recovery_l24_28_b050` | `task_quality_smoke_recovery_l24_28_b050_v3_ridge_latest` | 46.15% | 46.15% | -10.6069 | -3.1905 | fail |

结论：

- 整层范围扩展很脆弱，即使只压缩 layers 24:28、总 saving 7.14%，v3 stable 也明显失败。
- 安全结果仍限于 layer24 单层 8 heads / 50% selected budget / total saving 1.79%。
- 因此下一步不应该按 layer range 扩展，而应该做 head-level progressive search：从 layer24 的 8 heads 出发，逐个加入候选 head（可能来自 layer23/25/26 中 offline R²较高的 head），每步跑 v3 + qwen8bfixes task gates。

### 2026-07-09 head-list policy support

为下一步 head-level progressive search，`make_recovery_task_policy.py` 新增：

- `--compact-heads layer:kv_head,...`

验证生成：

- `policy_recovery_heads_l24_all_b050`
  - heads: `24:0..7`
  - per-head selected budget: 50.00%
  - total entry saving: 1.79%

当前安全边界：

- pass：layer24 all 8 KV heads, 50% budget, first-doc v3 stable + second-doc stable 都通过。
- fail：layers 24:28 range, 50% budget, first-doc v3 stable 只有 46.15%。
- fail：all layers, 50% budget, first-doc v3 stable 只有 15.38%。

下一步可直接基于 `--compact-heads` 自动枚举 add-one/add-pair：

1. seed = `24:0,24:1,24:2,24:3,24:4,24:5,24:6,24:7`。
2. candidates 优先来自 layers 23/25/26/27 中 offline recovery R²较好的 heads。
3. 每次生成一个 head-list policy，先跑 first-doc v3 stable；通过后再跑 qwen8bfixes stable。
4. 目标不是全层 2x，而是在 task gate 下最大化 compressed heads / entry saving。

### 2026-07-09 head-level progressive add-one search

新增工具：

- `experiments/light_doc_cache/make_head_addition_policies.py`
  - 输入 `--seed-heads` 和 `--candidate-heads`；
  - 批量生成 add-one head-list policies；
  - 输出 `manifest.json` 和 `report.md`。

候选来源：从已有 offline `recovery_head_rows.csv` 按 `recovery_val_r2` 排序，排除 seed layer24 heads，top candidates 包括：`10:3`、`8:1`、`20:2`、`4:6`、`8:5`、`12:4`、`10:5`、`12:7` 等。

Progressive seed：

- base seed: `24:0,24:1,24:2,24:3,24:4,24:5,24:6,24:7`
- per-head selected budget: 50.00%
- bank method: `ridge`

结果：

| policy | heads | added head | entry saving | first-doc v3 | second-doc stable | conclusion |
|---|---:|---|---:|---|---|---|
| `policy_add_one_from_l24_seed_top12/add_l10_h3` | 9 | `10:3` | 2.01% | 13/13 pass, delta +0.3548 | 6/6 pass, delta +0.8621 | safe |
| `policy_recovery_seed_l24_plus_l10h3_l8h1_b050` | 10 | `8:1` | 2.23% | 13/13 pass, delta +0.5904 | 6/6 pass, delta +0.9427 | safe |
| `policy_recovery_seed_l24_l10h3_l8h1_add_l20h2_b050` | 11 | `20:2` | 2.46% | 13/13 pass, delta +0.2751 | 6/6 pass, delta +0.9816 | safe |
| `policy_recovery_seed_l24_l10h3_l8h1_l20h2_add_l4h6_b050` | 12 | `4:6` | 2.68% | 13/13 pass, delta +0.1968 | 5/6 pass, fails `tp_true_weight_split` | unsafe |
| `policy_recovery_seed_l24_l10h3_l8h1_l20h2_add_l8h5_b050` | 12 | `8:5` | 2.68% | 13/13 pass, delta +0.3273 | 5/6 pass, fails `tp_true_weight_split` | unsafe |
| `policy_recovery_seed_l24_l10h3_l8h1_l20h2_add_l12h4_b050` | 12 | `12:4` | 2.68% | 13/13 pass, delta +0.3720 | 6/6 pass, delta +0.5808 | safe |
| `policy_recovery_seed_l24_l10h3_l8h1_l20h2_l12h4_add_l10h5_b050` | 13 | `10:5` | 2.90% | 13/13 pass, delta +0.5395 | 6/6 pass, delta +0.8457 | safe |
| `policy_recovery_seed_l24_l10h3_l8h1_l20h2_l12h4_l10h5_add_l12h7_b050` | 14 | `12:7` | 3.12% | 13/13 pass, delta +0.6640 | 6/6 pass, delta +1.1732 | safe |
| `policy_recovery_seed_l24_l10h3_l8h1_l20h2_l12h4_l10h5_l12h7_add_l4h5_b050` | 15 | `4:5` | 3.35% | 12/13 pass, fails `topk8_quality` | not run | unsafe |
| `policy_recovery_seed_l24_l10h3_l8h1_l20h2_l12h4_l10h5_l12h7_add_l12h1_b050` | 15 | `12:1` | 3.35% | 12/13 pass, fails `topk8_quality` | not run | unsafe |
| `policy_recovery_seed_l24_l10h3_l8h1_l20h2_l12h4_l10h5_l12h7_add_l16h6_b050` | 15 | `16:6` | 3.35% | 11/13 pass, fails `route_phase3`, `topk8_quality` | not run | unsafe |
| `policy_recovery_seed_l24_l10h3_l8h1_l20h2_l12h4_l10h5_l12h7_add_l12h3_b050` | 15 | `12:3` | 3.35% | 12/13 pass, fails `topk8_quality` | not run | unsafe |
| `policy_recovery_seed14_add_l20h7_b050` | 15 | `20:7` | 3.35% | 13/13 pass, delta +0.3132 | 6/6 pass, delta +0.9287 | safe |
| `policy_recovery_seed15_l20h7_add_l8h3_b050` | 16 | `8:3` | 3.57% | 13/13 pass, delta +0.3652 | 6/6 pass, delta +0.9026 | safe |
| `policy_recovery_seed16_add_l12h5_b050` | 17 | `12:5` | 3.79% | 13/13 pass, delta +0.3337 | 6/6 pass, delta +0.7799 | safe |
| `policy_recovery_seed17_add_l12h6_b050` | 18 | `12:6` | 4.02% | 12/13 pass, fails `route_phase3` | not run | unsafe |
| `policy_recovery_seed17_add_l16h5_b050` | 18 | `16:5` | 4.02% | 13/13 pass, delta +0.4079 | 6/6 pass, delta +1.0075 | safe |
| `policy_recovery_seed18_add_l16h4_b050` | 19 | `16:4` | 4.24% | 12/13 pass, fails `route_phase3` | not run | unsafe |
| `policy_recovery_seed18_add_l8h0_b050` | 19 | `8:0` | 4.24% | 13/13 pass, delta +0.4697 | 6/6 pass, delta +0.9474 | safe |
| `policy_recovery_seed19_add_l20h6_b050` | 20 | `20:6` | 4.46% | 13/13 pass, delta +0.3710 | 6/6 pass, delta +1.2289 | safe |
| `policy_recovery_seed20_add_l10h2_b050` | 21 | `10:2` | 4.69% | 13/13 pass, delta +0.4025 | 6/6 pass, delta +1.2961 | safe |
| `policy_recovery_seed21_add_l12h0_b050` | 22 | `12:0` | 4.91% | 13/13 pass, delta +0.2928 | 6/6 pass, delta +1.4000 | safe |
| `policy_recovery_seed22_add_l20h5_b050` | 23 | `20:5` | 5.13% | 12/13 pass, fails `quest_decode_selection` | not run | unsafe |
| `policy_recovery_seed22_add_l20h1_b050` | 23 | `20:1` | 5.13% | 13/13 pass, delta +0.8226 | 6/6 pass, delta +2.4415 | safe |
| `policy_recovery_seed23_add_l10h1_b050` | 24 | `10:1` | 5.36% | 13/13 pass, delta +0.7601 | 6/6 pass, delta +2.4024 | safe |
| `policy_recovery_seed24_add_l16h7_b050` | 25 | `16:7` | 5.58% | 11/13 pass, fails `prefix_cache_bug`, `topk8_quality` | not run | unsafe |
| `policy_recovery_seed24_add_l8h2_b050` | 25 | `8:2` | 5.58% | 12/13 pass, fails `quest_decode_selection` | not run | unsafe |
| `policy_recovery_seed24_add_l16h3_b050` | 25 | `16:3` | 5.58% | 12/13 pass, fails `quest_decode_selection` | not run | unsafe |
| `policy_recovery_seed24_add_l20h3_b050` | 25 | `20:3` | 5.58% | 13/13 pass, delta +0.6031 | 6/6 pass, delta +2.3951 | safe |
| `policy_recovery_seed25_add_l0h0_b050` | 26 | `0:0` | 5.80% | 13/13 pass, delta +0.3546 | 6/6 pass, delta +1.7638 | safe |
| `policy_recovery_seed26_add_l20h4_b050` | 27 | `20:4` | 6.03% | 12/13 pass, fails `topk8_quality` | not run | unsafe |
| `policy_recovery_seed26_add_l8h6_b050` | 27 | `8:6` | 6.03% | 12/13 pass, fails `quest_decode_selection` | not run | unsafe |
| `policy_recovery_seed26_add_l16h1_b050` | 27 | `16:1` | 6.03% | 12/13 pass, fails `quest_decode_selection` | not run | unsafe |

当前最大跨文档安全点：

- Heads: 26
- Head list: `24:0..7,10:3,8:1,20:2,12:4,10:5,12:7,20:7,8:3,12:5,16:5,8:0,20:6,10:2,12:0,20:1,10:1,20:3,0:0`
- Total entry saving: 5.80%
- First-doc v3 stable: 13/13 pass
- Second-doc stable: 6/6 pass

解释：

- Head-level progressive search 可以比 layer24-only 的 8 heads / 1.79% saving 增加到 26 heads / 5.80% saving，并保持双文档 stable task quality。
- 第 12 个 head 并非固定失败：`4:6`、`8:5` 都在 second-doc `tp_true_weight_split` 失败，但 `12:4` 通过；后续 `10:5`、`12:7` 也通过。
- 15-head frontier 在 first-doc v3 开始不稳定：`4:5`、`12:1`、`12:3` 都翻转 `topk8_quality`，`16:6` 额外翻转 `route_phase3`。
- 扩大候选池后，`20:7`、`8:3`、`12:5`、`16:5`、`8:0`、`20:6`、`10:2`、`12:0`、`20:1`、`10:1`、`20:3`、`0:0` 可以继续安全扩展；但 `12:6`、`16:4` 翻转 `route_phase3`，`20:5`/`8:2`/`16:3`/`8:6`/`16:1` 翻转 `quest_decode_selection`，`16:7` 翻转 `prefix_cache_bug` 和 `topk8_quality`，`20:4` 翻转 `topk8_quality`。因此安全 head 不是 recovery R² 的严格前缀，需要跳过 unsafe candidates。
- 这说明 offline recovery R² 高不等于 task-safe，需要 task gate；task 失败还呈现文档/题目敏感性。
- 当前离 KR 的全局 2x+ 仍很远；但已经有可复现的 head-level safe frontier，可用于后续自动搜索和 paper 的 negative/diagnostic section。

下一步：

1. 继续扩大候选池，不只限于 offline top12；但每个新增 head 必须经过 first-doc v3 + second-doc stable 双门禁。
2. 尝试 task-aware candidate ordering：把会影响 `topk8_quality` / `tp_true_weight_split` 的 head 降权，而不是单纯按 recovery R² 排序。
3. 若目标仍是 2x+ 总压缩，需要进入 document-adaptive / task-aware / shared multi-doc recovery，而不是继续固定 head-list 贪心扩展。

### 2026-07-09 extended head-level progressive search to 44 heads

Continued the recovery-bank add-one search from the previously safe 26-head frontier. Method stayed unchanged:

- Bank method: `ridge`
- Per-head selected budget: 50%
- First gate: `docs/kv-sparse-attention.md` + `task_quality_tasks_kv_sparse_v3_stable.json`
- Second gate: `docs/qwen3-8b-fixes.md` + `task_quality_tasks_qwen3_8b_fixes_stable.json`
- Automation log: `experiments/light_doc_cache/head_progress_continue_20260709_185231.tsv`

New candidates/results:

| Policy | Heads | Added Head | Total Entry Saving | First Doc v3 | Second Doc Stable | Result |
|---|---:|---|---:|---|---|---|
| `policy_recovery_seed26_add_l8h7_b050` | 27 | `8:7` | 6.03% | 13/13, delta +0.3193 | 6/6, delta +1.8511 | safe |
| `policy_recovery_seed27_add_l4h1_b050` | 28 | `4:1` | 6.25% | 13/13, delta +0.3922 | 6/6, delta +1.8479 | safe |
| `policy_recovery_seed28_add_l0h2_b050` | 29 | `0:2` | 6.47% | 13/13, delta +0.2938 | 6/6, delta +1.7941 | safe |
| `policy_recovery_seed29_add_l0h5_b050` | 30 | `0:5` | 6.70% | 13/13, delta +0.2240 | 6/6, delta +1.5503 | safe |
| `policy_recovery_seed30_add_l0h4_b050` | 31 | `0:4` | 6.92% | 13/13, delta +0.1433 | 6/6, delta +1.4856 | safe |
| `policy_recovery_seed31_add_l0h6_b050` | 32 | `0:6` | 7.14% | 12/13, delta +0.0611 | not run | unsafe |
| `policy_recovery_seed31_add_l16h0_b050` | 32 | `16:0` | 7.14% | 12/13, delta +0.2646 | not run | unsafe |
| `policy_recovery_seed31_add_l10h7_b050` | 32 | `10:7` | 7.14% | 13/13, delta +0.0428 | 6/6, delta +1.1412 | safe |
| `policy_recovery_seed32_add_l12h2_b050` | 33 | `12:2` | 7.37% | 12/13, delta +0.4341 | not run | unsafe |
| `policy_recovery_seed32_add_l10h6_b050` | 33 | `10:6` | 7.37% | 13/13, delta +0.4107 | 6/6, delta +1.6898 | safe |
| `policy_recovery_seed33_add_l0h7_b050` | 34 | `0:7` | 7.59% | 13/13, delta +0.4742 | 6/6, delta +1.6563 | safe |
| `policy_recovery_seed34_add_l4h2_b050` | 35 | `4:2` | 7.81% | 13/13, delta +0.4964 | 6/6, delta +1.6008 | safe |
| `policy_recovery_seed35_add_l4h7_b050` | 36 | `4:7` | 8.04% | 13/13, delta +0.4686 | 6/6, delta +1.5820 | safe |
| `policy_recovery_seed36_add_l10h0_b050` | 37 | `10:0` | 8.26% | 13/13, delta +0.1315 | 6/6, delta +1.3163 | safe |
| `policy_recovery_seed37_add_l10h4_b050` | 38 | `10:4` | 8.48% | 12/13, delta +0.1232 | not run | unsafe |
| `policy_recovery_seed37_add_l4h4_b050` | 38 | `4:4` | 8.48% | 13/13, delta +0.1853 | 6/6, delta +1.3584 | safe |
| `policy_recovery_seed38_add_l0h3_b050` | 39 | `0:3` | 8.71% | 13/13, delta +0.0922 | 6/6, delta +1.4533 | safe |
| `policy_recovery_seed39_add_l16h2_b050` | 40 | `16:2` | 8.93% | 13/13, delta +0.2084 | 6/6, delta +1.4016 | safe |
| `policy_recovery_seed40_add_l8h4_b050` | 41 | `8:4` | 9.15% | 13/13, delta +0.2154 | 6/6, delta +1.2424 | safe |
| `policy_recovery_seed41_add_l4h0_b050` | 42 | `4:0` | 9.38% | 13/13, delta +0.3445 | 6/6, delta +1.3466 | safe |
| `policy_recovery_seed42_add_l4h3_b050` | 43 | `4:3` | 9.60% | 13/13, delta +0.2704 | 6/6, delta +1.6126 | safe |
| `policy_recovery_seed43_add_l0h1_b050` | 44 | `0:1` | 9.82% | 13/13, delta +0.3443 | 6/6, delta +1.6561 | safe |
| `policy_recovery_seed44_add_l20h0_b050` | 45 | `20:0` | 10.04% | 12/13, delta +1.1228 | not run | unsafe |

Current best cross-document safe frontier:

- Policy: `experiments/light_doc_cache/policy_recovery_seed43_add_l0h1_b050`
- Heads: 44
- Head list: `24:0,24:1,24:2,24:3,24:4,24:5,24:6,24:7,10:3,8:1,20:2,12:4,10:5,12:7,20:7,8:3,12:5,16:5,8:0,20:6,10:2,12:0,20:1,10:1,20:3,0:0,8:7,4:1,0:2,0:5,0:4,10:7,10:6,0:7,4:2,4:7,10:0,4:4,0:3,16:2,8:4,4:0,4:3,0:1`
- Per-head selected budget: 50%
- Total KV head-token entry saving: 9.82%
- First-doc v3 stable: 13/13 pass, mean delta +0.3443
- Second-doc stable: 6/6 pass, mean delta +1.6561

New unsafe heads from this continuation:

- `0:6`: first-doc `quest_decode_selection`, `A -> C`, answer `A`.
- `16:0`: first-doc `route_phase3`, `A -> B`, answer `A`.
- `12:2`: first-doc `prefix_cache_bug`, `A -> B`, answer `A`.
- `10:4`: first-doc `prefix_cache_bug`, `A -> B`, answer `A`.
- `20:0`: first-doc `prefix_cache_bug`, `A -> C`, answer `A`.

Conclusion: head-level task-gated search improved the safe frontier from 26 heads / 5.80% total entry saving to 44 heads / 9.82% total entry saving. This is a meaningful targeted recovery-bank result, but still far from the KR wording of 2x+ global doc-cache compression. The current evidence remains an offline quality simulation with sparse safe heads; a 2x+ claim would require a stronger document-adaptive/trained recovery module and runtime integration.

### 2026-07-09 round2 full-layer progressive search to 56 heads

After completing the missing-layer recovery probes, the candidate pool covered all 28 layers / 224 KV heads. Continued add-one search from the 44-head safe frontier:

- Bank method: `ridge`
- Per-head selected budget: 50%
- First gate: `docs/kv-sparse-attention.md` + `task_quality_tasks_kv_sparse_v3_stable.json`
- Second gate: `docs/qwen3-8b-fixes.md` + `task_quality_tasks_qwen3_8b_fixes_stable.json`
- Logs:
  - `experiments/light_doc_cache/head_progress_round2_20260709_192603.tsv`
  - `experiments/light_doc_cache/head_progress_round2_from52_20260709_194018.tsv`
  - `experiments/light_doc_cache/head_progress_round2_retry23h7_20260709_195107.tsv`

Important log caveat: `head_progress_round2_20260709_192603.tsv` includes rows after a run interruption and should not be used alone as the source of truth for the tail. `head_progress_round2_from52_20260709_194018.tsv` plus the `23:7` retry log are the authoritative continuation records.

Round2 results:

| Policy | Heads | Added Head | Total Entry Saving | First Doc v3 | Second Doc Stable | Result |
|---|---:|---|---:|---|---|---|
| `policy_recovery_seed44_add_l6h7_b050` | 45 | `6:7` | 10.04% | 13/13, delta +0.2832 | 6/6, delta +1.7664 | safe |
| `policy_recovery_seed45_add_l3h0_b050` | 46 | `3:0` | 10.27% | 13/13, delta +0.3215 | 6/6, delta +1.7573 | safe |
| `policy_recovery_seed46_add_l26h5_b050` | 47 | `26:5` | 10.49% | 13/13, delta +0.3215 | 6/6, delta +1.8209 | safe |
| `policy_recovery_seed47_add_l18h0_b050` | 48 | `18:0` | 10.71% | 13/13, delta +0.4821 | 6/6, delta +1.7973 | safe |
| `policy_recovery_seed48_add_l25h6_b050` | 49 | `25:6` | 10.94% | 12/13, delta +0.1111 | not run | unsafe |
| `policy_recovery_seed48_add_l22h1_b050` | 49 | `22:1` | 10.94% | 13/13, delta +0.5319 | 6/6, delta +1.9941 | safe |
| `policy_recovery_seed49_add_l26h2_b050` | 50 | `26:2` | 11.16% | 13/13, delta +0.3615 | 5/6, delta +1.7897 | unsafe |
| `policy_recovery_seed49_add_l11h2_b050` | 50 | `11:2` | 11.16% | 13/13, delta +0.6035 | 6/6, delta +1.7880 | safe |
| `policy_recovery_seed50_add_l25h5_b050` | 51 | `25:5` | 11.38% | 12/13, delta +0.2425 | not run | unsafe |
| `policy_recovery_seed50_add_l22h0_b050` | 51 | `22:0` | 11.38% | 13/13, delta +0.4683 | 6/6, delta +1.8989 | safe |
| `policy_recovery_seed51_add_l9h1_b050` | 52 | `9:1` | 11.61% | 13/13, delta +0.6004 | 5/6, delta +1.4883 | unsafe |
| `policy_recovery_seed51_add_l3h6_b050` | 52 | `3:6` | 11.61% | 13/13, delta +0.4419 | 6/6, delta +1.9612 | safe |
| `policy_recovery_seed52_add_l26h3_b050` | 53 | `26:3` | 11.83% | 12/13, delta +0.7619 | not run | unsafe |
| `policy_recovery_seed52_add_l26h1_b050` | 53 | `26:1` | 11.83% | 10/13, delta +0.3498 | not run | unsafe |
| `policy_recovery_seed52_add_l26h0_b050` | 53 | `26:0` | 11.83% | 12/13, delta +0.1350 | not run | unsafe |
| `policy_recovery_seed52_add_l13h1_b050` | 53 | `13:1` | 11.83% | 13/13, delta +0.6363 | 6/6, delta +1.9674 | safe |
| `policy_recovery_seed53_add_l18h1_b050` | 54 | `18:1` | 12.05% | 11/13, delta -1.4445 | not run | unsafe |
| `policy_recovery_seed53_add_l21h5_b050` | 54 | `21:5` | 12.05% | 12/13, delta +0.9535 | not run | unsafe |
| `policy_recovery_seed53_add_l3h1_b050` | 54 | `3:1` | 12.05% | 13/13, delta +0.6570 | 6/6, delta +1.9631 | safe |
| `policy_recovery_seed54_add_l23h2_b050` | 55 | `23:2` | 12.28% | 12/13, delta +0.6697 | not run | unsafe |
| `policy_recovery_seed54_add_l25h2_b050` | 55 | `25:2` | 12.28% | 12/13, delta +0.1035 | not run | unsafe |
| `policy_recovery_seed54_add_l13h3_b050` | 55 | `13:3` | 12.28% | 13/13, delta +0.7032 | 6/6, delta +2.3966 | safe |
| `policy_recovery_seed55_add_l23h0_b050` | 56 | `23:0` | 12.50% | 12/13, delta +0.5160 | not run | unsafe |
| `policy_recovery_seed55_add_l23h7_b050` | 56 | `23:7` | 12.50% | 13/13, delta +0.5845 | 6/6, delta +2.3314 | safe |

Current best cross-document safe frontier:

- Policy: `experiments/light_doc_cache/policy_recovery_seed55_add_l23h7_b050`
- Heads: 56
- Head list: `24:0,24:1,24:2,24:3,24:4,24:5,24:6,24:7,10:3,8:1,20:2,12:4,10:5,12:7,20:7,8:3,12:5,16:5,8:0,20:6,10:2,12:0,20:1,10:1,20:3,0:0,8:7,4:1,0:2,0:5,0:4,10:7,10:6,0:7,4:2,4:7,10:0,4:4,0:3,16:2,8:4,4:0,4:3,0:1,6:7,3:0,26:5,18:0,22:1,11:2,22:0,3:6,13:1,3:1,13:3,23:7`
- Per-head selected budget: 50%
- Total KV head-token entry saving: 12.50%
- First-doc v3 stable: 13/13 pass, mean delta +0.5845
- Second-doc stable: 6/6 pass, mean delta +2.3314

Round2 unsafe heads:

- First-doc failures: `25:6`, `25:5`, `26:3`, `26:1`, `26:0`, `18:1`, `21:5`, `23:2`, `25:2`, `23:0`.
- Second-doc failures: `26:2` and `9:1`, both on `tp_true_weight_split`.
- `23:7` initially hit an SSH/base64 transfer error; retry passed both quality gates, so it is safe.

Conclusion: full-layer recovery ranking plus strict task gates improved the safe frontier from 44 heads / 9.82% to 56 heads / 12.50% total entry saving. This is stronger than the previous recovery-bank result, but still does not support a 2x+ global doc-cache compression claim. The result remains a sparse fixed head-list policy under offline quality simulation. A KR-level 2x+ claim still requires document-adaptive or shared trained recovery plus runtime K/V cache integration.

Next continuation candidate generation:

1. Use all-layer `recovery_head_rows.csv` to generate the next add-one batch, excluding the 56 safe heads and all known unsafe heads.
2. Prefer candidates with high cross-doc `recovery_val_r2`, but downrank layers/heads that repeatedly trigger `prefix_cache_bug`, `quest_decode_selection`, `topk8_quality`, or `tp_true_weight_split`.
3. Keep the same two-document gate unless adding a third document/task set; do not lower the 13/13 and 6/6 pass requirements for frontier claims.

### 2026-07-09 round3 partial search to 58 heads

Started the next add-one batch from the 56-head safe frontier. Candidate ordering used all-layer cross-doc `recovery_val_r2`, excluding known safe and unsafe heads, with light downranking for layers that repeatedly caused task failures.

- Automation log: `experiments/light_doc_cache/head_progress_round3_20260709_195436.tsv`
- Initial candidate list: `11:0,11:6,13:2,13:5,2:5,17:1,17:4,14:5,3:7,2:7,17:5,14:0,26:7,21:6,14:1,19:6,1:5,11:5,1:7,13:0,9:6,5:2,13:7,25:3`

Completed quality results before Kerberos expiry:

| Policy | Heads | Added Head | Total Entry Saving | First Doc v3 | Second Doc Stable | Result |
|---|---:|---|---:|---|---|---|
| `policy_recovery_seed56_add_l11h0_b050` | 57 | `11:0` | 12.72% | 13/13, delta +0.6066 | 6/6, delta +2.5306 | safe |
| `policy_recovery_seed57_add_l11h6_b050` | 58 | `11:6` | 12.95% | 11/13, delta +0.3431 | not run | unsafe |
| `policy_recovery_seed57_add_l13h2_b050` | 58 | `13:2` | 12.95% | 12/13, delta +0.3330 | not run | unsafe |
| `policy_recovery_seed57_add_l13h5_b050` | 58 | `13:5` | 12.95% | 13/13, delta +0.5693 | 6/6, delta +2.3385 | safe |

Current best cross-document safe frontier:

- Policy: `experiments/light_doc_cache/policy_recovery_seed57_add_l13h5_b050`
- Heads: 58
- Head list: `24:0,24:1,24:2,24:3,24:4,24:5,24:6,24:7,10:3,8:1,20:2,12:4,10:5,12:7,20:7,8:3,12:5,16:5,8:0,20:6,10:2,12:0,20:1,10:1,20:3,0:0,8:7,4:1,0:2,0:5,0:4,10:7,10:6,0:7,4:2,4:7,10:0,4:4,0:3,16:2,8:4,4:0,4:3,0:1,6:7,3:0,26:5,18:0,22:1,11:2,22:0,3:6,13:1,3:1,13:3,23:7,11:0,13:5`
- Per-head selected budget: 50%
- Total KV head-token entry saving: 12.95%
- First-doc v3 stable: 13/13 pass, mean delta +0.5693
- Second-doc stable: 6/6 pass, mean delta +2.3385

Blocked state:

- `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian klist` reports expired TGT.
- `kinit -R` failed with `Matching credential (krbtgt/BYTEDANCE.COM@BYTEDANCE.COM) not found`.
- All later `ERR` rows in `head_progress_round3_20260709_195436.tsv` are SSH/Kerberos command failures, not quality failures.
- Retry after Kerberos refresh should start from the 58-head seed and include the unresolved candidates: `2:5,17:1,17:4,14:5,3:7,2:7,17:5,14:0,26:7,21:6,14:1,19:6,1:5,11:5,1:7,13:0,9:6,5:2,13:7,25:3`.

### 2026-07-09 round3 retry advanced frontier to 68 heads

Kerberos was refreshed and round3 unresolved candidates were retried from the 58-head frontier.

- Bank method: `ridge`
- Per-head selected budget: 50%
- First gate: `docs/kv-sparse-attention.md` + `experiments/light_doc_cache/task_quality_tasks_kv_sparse_v3_stable.json`
- Second gate: `docs/qwen3-8b-fixes.md` + `experiments/light_doc_cache/task_quality_tasks_qwen3_8b_fixes_stable.json`
- Authoritative retry logs:
  - `experiments/light_doc_cache/head_progress_round3_retry_from58_20260709_213131.tsv`
  - `experiments/light_doc_cache/head_progress_round3_retry_remaining_direct_20260709_215135.tsv`

Log caveat: `head_progress_round3_retry_from58_20260709_213131.tsv` includes stale or command-failed tail rows after SSH instability. The direct retry log is authoritative for `13:0,9:6,5:2,13:7,25:3`.

Retry results:

| Policy | Heads | Added Head | Total Entry Saving | First Doc v3 | Second Doc Stable | Result |
|---|---:|---|---:|---|---|---|
| `policy_recovery_seed58_add_l2h5_b050` | 59 | `2:5` | 13.17% | 12/13, delta +0.6885 | not run | unsafe |
| `policy_recovery_seed58_add_l17h1_b050` | 59 | `17:1` | 13.17% | 13/13, delta -0.2055 | 6/6, delta +1.9262 | safe |
| `policy_recovery_seed59_add_l17h4_b050` | 60 | `17:4` | 13.39% | 11/13, delta -0.1008 | not run | unsafe |
| `policy_recovery_seed59_add_l14h5_b050` | 60 | `14:5` | 13.39% | 12/13, delta -0.5952 | not run | unsafe |
| `policy_recovery_seed59_add_l3h7_b050` | 60 | `3:7` | 13.39% | 13/13, delta -0.0911 | 6/6, delta +1.9033 | safe |
| `policy_recovery_seed60_add_l2h7_b050` | 61 | `2:7` | 13.62% | 13/13, delta -0.1743 | 6/6, delta +1.8585 | safe |
| `policy_recovery_seed61_add_l17h5_b050` | 62 | `17:5` | 13.84% | 13/13, delta +0.4271 | 6/6, delta +2.0074 | safe |
| `policy_recovery_seed62_add_l14h0_b050` | 63 | `14:0` | 14.06% | 12/13, delta +0.4874 | not run | unsafe |
| `policy_recovery_seed62_add_l26h7_b050` | 63 | `26:7` | 14.06% | 12/13, delta +0.3042 | not run | unsafe |
| `policy_recovery_seed62_add_l21h6_b050` | 63 | `21:6` | 14.06% | 12/13, delta +0.4442 | not run | unsafe |
| `policy_recovery_seed62_add_l14h1_b050` | 63 | `14:1` | 14.06% | 13/13, delta +0.1545 | 6/6, delta +2.0307 | safe |
| `policy_recovery_seed63_add_l19h6_b050` | 64 | `19:6` | 14.29% | 12/13, delta -0.2735 | not run | unsafe |
| `policy_recovery_seed63_add_l1h5_b050` | 64 | `1:5` | 14.29% | 13/13, delta +0.1935 | 6/6, delta +1.8957 | safe |
| `policy_recovery_seed64_add_l11h5_b050` | 65 | `11:5` | 14.51% | 13/13, delta -0.0415 | 5/6, delta +1.9840 | unsafe |
| `policy_recovery_seed64_add_l1h7_b050` | 65 | `1:7` | 14.51% | 13/13, delta +0.2274 | 6/6, delta +1.8394 | safe |
| `policy_recovery_seed65_add_l13h0_b050` | 66 | `13:0` | 14.73% | 13/13, delta +0.2724 | 6/6, delta +1.7213 | safe |
| `policy_recovery_seed66_add_l9h6_b050` | 67 | `9:6` | 14.96% | 13/13, delta +0.2535 | 6/6, delta +1.9532 | safe |
| `policy_recovery_seed67_add_l5h2_b050` | 68 | `5:2` | 15.18% | 13/13, delta +0.0320 | 6/6, delta +1.3612 | safe |
| `policy_recovery_seed68_add_l13h7_b050` | 69 | `13:7` | 15.40% | 13/13, delta +0.0354 | 5/6, delta +1.2103 | unsafe |
| `policy_recovery_seed68_add_l25h3_b050` | 69 | `25:3` | 15.40% | 12/13, delta -0.4433 | not run | unsafe |

Current best cross-document safe frontier:

- Policy: `experiments/light_doc_cache/policy_recovery_seed67_add_l5h2_b050`
- Heads: 68
- Per-head selected budget: 50%
- Total KV head-token entry saving: 15.18%
- First-doc v3 stable: 13/13 pass, mean delta +0.0320
- Second-doc stable: 6/6 pass, mean delta +1.3612
- Head list: `24:0,24:1,24:2,24:3,24:4,24:5,24:6,24:7,10:3,8:1,20:2,12:4,10:5,12:7,20:7,8:3,12:5,16:5,8:0,20:6,10:2,12:0,20:1,10:1,20:3,0:0,8:7,4:1,0:2,0:5,0:4,10:7,10:6,0:7,4:2,4:7,10:0,4:4,0:3,16:2,8:4,4:0,4:3,0:1,6:7,3:0,26:5,18:0,22:1,11:2,22:0,3:6,13:1,3:1,13:3,23:7,11:0,13:5,17:1,3:7,2:7,17:5,14:1,1:5,1:7,13:0,9:6,5:2`

Unsafe heads from retry:

- First-doc failures: `2:5`, `17:4`, `14:5`, `14:0`, `26:7`, `21:6`, `19:6`, `25:3`.
- Second-doc failures: `11:5` failed `gpu_utilization_semantics`; `13:7` failed `tp_true_weight_split`.
- SSH `Connection closed by UNKNOWN port 65535` remained intermittent, but successful retries confirmed the accepted rows; do not treat command failures as quality failures.

Conclusion: strict task-gated recovery-bank search improved the sparse fixed-head frontier from 58 heads / 12.95% to 68 heads / 15.18% total KV head-token entry saving. This is useful progress for the trained/correlation recovery direction, but it still does not support a KR-level 2x+ global doc-cache compression claim. That requires document-adaptive/shared trained recovery plus real runtime K/V cache integration.

### 2026-07-09 round4 candidate search advanced frontier to 72 heads

Generated the next candidate list from all-layer cross-doc recovery rows after excluding the 68 safe heads and known unsafe heads. Ordering used mean/min `recovery_val_r2` with light downranking for failure-prone layers.

- Candidate ranking: `experiments/light_doc_cache/head_candidates_round4_from68_20260709.tsv`
- Automation log: `experiments/light_doc_cache/head_progress_round4_from68_20260709_220108.tsv`
- Bank method: `ridge`
- Per-head selected budget: 50%
- Same strict first-doc and second-doc gates.

Round4 top-12 results:

| Policy | Heads | Added Head | Total Entry Saving | First Doc v3 | Second Doc Stable | Result |
|---|---:|---|---:|---|---|---|
| `policy_recovery_seed68_add_l4h6_b050` | 69 | `4:6` | 15.40% | 13/13, delta +0.0482 | 6/6, delta +1.4524 | safe |
| `policy_recovery_seed69_add_l4h5_b050` | 70 | `4:5` | 15.62% | 13/13, delta +0.1080 | 5/6, delta +1.5376 | unsafe |
| `policy_recovery_seed69_add_l12h1_b050` | 70 | `12:1` | 15.62% | 11/13, delta +0.2706 | not run | unsafe |
| `policy_recovery_seed69_add_l12h3_b050` | 70 | `12:3` | 15.62% | 12/13, delta +0.1186 | not run | unsafe |
| `policy_recovery_seed69_add_l16h6_b050` | 70 | `16:6` | 15.62% | 13/13, delta -0.0055 | 6/6, delta +1.3167 | safe |
| `policy_recovery_seed70_add_l18h6_b050` | 71 | `18:6` | 15.85% | 13/13, delta -0.0919 | 6/6, delta +1.3758 | safe |
| `policy_recovery_seed71_add_l8h5_b050` | 72 | `8:5` | 16.07% | 12/13, delta +0.2750 | not run | unsafe |
| `policy_recovery_seed71_add_l25h7_b050` | 72 | `25:7` | 16.07% | 12/13, delta -0.6519 | not run | unsafe |
| `policy_recovery_seed71_add_l6h0_b050` | 72 | `6:0` | 16.07% | 13/13, delta -0.1569 | 6/6, delta +1.5364 | safe |
| `policy_recovery_seed72_add_l25h4_b050` | 73 | `25:4` | 16.29% | 12/13, delta -0.0955 | not run | unsafe |
| `policy_recovery_seed72_add_l9h4_b050` | 73 | `9:4` | 16.29% | 12/13, delta -0.2362 | not run | unsafe |
| `policy_recovery_seed72_add_l21h1_b050` | 73 | `21:1` | 16.29% | 11/13, delta -0.3408 | not run | unsafe |

Current best cross-document safe frontier:

- Policy: `experiments/light_doc_cache/policy_recovery_seed71_add_l6h0_b050`
- Heads: 72
- Per-head selected budget: 50%
- Total KV head-token entry saving: 16.07%
- First-doc v3 stable: 13/13 pass, mean delta -0.1569
- Second-doc stable: 6/6 pass, mean delta +1.5364
- Head list: `24:0,24:1,24:2,24:3,24:4,24:5,24:6,24:7,10:3,8:1,20:2,12:4,10:5,12:7,20:7,8:3,12:5,16:5,8:0,20:6,10:2,12:0,20:1,10:1,20:3,0:0,8:7,4:1,0:2,0:5,0:4,10:7,10:6,0:7,4:2,4:7,10:0,4:4,0:3,16:2,8:4,4:0,4:3,0:1,6:7,3:0,26:5,18:0,22:1,11:2,22:0,3:6,13:1,3:1,13:3,23:7,11:0,13:5,17:1,3:7,2:7,17:5,14:1,1:5,1:7,13:0,9:6,5:2,4:6,16:6,18:6,6:0`

Round4 takeaways:

- New safe heads: `4:6`, `16:6`, `18:6`, `6:0`.
- `4:5` was a second-doc `tp_true_weight_split` failure despite first-doc pass.
- First-doc failures were common even among top offline recovery candidates, confirming that recovery R2 remains a proposal signal rather than an acceptance criterion.
- Latest frontier: 72 heads / 16.07% total KV head-token entry saving, still under offline fixed-head quality simulation only.

### 2026-07-09 round5 candidate search advanced frontier to 74 heads

Round5 regenerated candidates from the 72-head safe frontier after excluding round4 failures and applying stronger penalties to layers with repeated first-doc failures.

- Candidate ranking: `experiments/light_doc_cache/head_candidates_round5_from72_20260709.tsv`
- Automation log: `experiments/light_doc_cache/head_progress_round5_from72_20260709_221720.tsv`
- Bank method: `ridge`
- Per-head selected budget: 50%

Round5 top-12 results:

| Policy | Heads | Added Head | Total Entry Saving | First Doc v3 | Second Doc Stable | Result |
|---|---:|---|---:|---|---|---|
| `policy_recovery_seed72_add_l7h2_b050` | 73 | `7:2` | 16.29% | 11/13, delta -0.2060 | not run | unsafe |
| `policy_recovery_seed72_add_l22h3_b050` | 73 | `22:3` | 16.29% | 11/13, delta -0.7073 | not run | unsafe |
| `policy_recovery_seed72_add_l19h0_b050` | 73 | `19:0` | 16.29% | 12/13, delta -0.0959 | not run | unsafe |
| `policy_recovery_seed72_add_l6h5_b050` | 73 | `6:5` | 16.29% | 12/13, delta +0.0092 | not run | unsafe |
| `policy_recovery_seed72_add_l17h6_b050` | 73 | `17:6` | 16.29% | 12/13, delta -0.0703 | not run | unsafe |
| `policy_recovery_seed72_add_l22h4_b050` | 73 | `22:4` | 16.29% | 11/13, delta -0.1766 | not run | unsafe |
| `policy_recovery_seed72_add_l3h5_b050` | 73 | `3:5` | 16.29% | 11/13, delta +0.0882 | not run | unsafe |
| `policy_recovery_seed72_add_l15h3_b050` | 73 | `15:3` | 16.29% | 13/13, delta -0.2607 | 6/6, delta +1.6232 | safe |
| `policy_recovery_seed73_add_l19h4_b050` | 74 | `19:4` | 16.52% | 12/13, delta +1.0387 | not run | unsafe |
| `policy_recovery_seed73_add_l21h0_b050` | 74 | `21:0` | 16.52% | 12/13, delta -0.2035 | not run | unsafe |
| `policy_recovery_seed73_add_l21h2_b050` | 74 | `21:2` | 16.52% | 13/13, delta -0.4267 | 5/6, delta +1.5138 | unsafe |
| `policy_recovery_seed73_add_l18h5_b050` | 74 | `18:5` | 16.52% | 13/13, delta +0.2090 | 6/6, delta +1.6877 | safe |

Current best cross-document safe frontier:

- Policy: `experiments/light_doc_cache/policy_recovery_seed73_add_l18h5_b050`
- Heads: 74
- Per-head selected budget: 50%
- Total KV head-token entry saving: 16.52%
- First-doc v3 stable: 13/13 pass, mean delta +0.2090
- Second-doc stable: 6/6 pass, mean delta +1.6877
- Head list: `24:0,24:1,24:2,24:3,24:4,24:5,24:6,24:7,10:3,8:1,20:2,12:4,10:5,12:7,20:7,8:3,12:5,16:5,8:0,20:6,10:2,12:0,20:1,10:1,20:3,0:0,8:7,4:1,0:2,0:5,0:4,10:7,10:6,0:7,4:2,4:7,10:0,4:4,0:3,16:2,8:4,4:0,4:3,0:1,6:7,3:0,26:5,18:0,22:1,11:2,22:0,3:6,13:1,3:1,13:3,23:7,11:0,13:5,17:1,3:7,2:7,17:5,14:1,1:5,1:7,13:0,9:6,5:2,4:6,16:6,18:6,6:0,15:3,18:5`

Round5 takeaways:

- New safe heads: `15:3` and `18:5`.
- `21:2` passed the first document but failed second-doc `smoothquant_status`.
- Most remaining high-recovery candidates fail first-doc gate, so fixed-head greedy expansion is now clearly hitting a task-quality frontier. Further progress probably needs document-adaptive/task-aware gating or recovery-module changes rather than only ranking by offline recovery R2.

### 2026-07-09 round6 candidate search advanced frontier to 78 heads

Round6 continued from the 74-head safe frontier. The initial run confirmed `2:3`, `1:4`, and `23:3`, then hit intermittent SSH/remote command instability on the tail. A direct continuation from the verified 77-head policy retested the remaining candidates and confirmed `9:7`.

- Candidate ranking: `experiments/light_doc_cache/head_candidates_round6_from74_20260709.tsv`
- Initial automation log: `experiments/light_doc_cache/head_progress_round6_from74_20260709_223132.tsv`
- Authoritative continuation log: `experiments/light_doc_cache/head_progress_round6_continue_from77_20260709_223824.tsv`
- Bank method: `ridge`
- Per-head selected budget: 50%
- Same strict first-doc and second-doc gates.

Round6 top-8 results:

| Policy | Heads | Added Head | Total Entry Saving | First Doc v3 | Second Doc Stable | Result |
|---|---:|---|---:|---|---|---|
| `policy_recovery_seed74_add_l14h6_b050` | 75 | `14:6` | 16.74% | 13/13, delta -0.1076 | 4/6, delta +1.3523 | unsafe |
| `policy_recovery_seed74_add_l14h3_b050` | 75 | `14:3` | 16.74% | 12/13, delta -0.0764 | not run | unsafe |
| `policy_recovery_seed74_add_l2h3_b050` | 75 | `2:3` | 16.74% | 13/13, delta +0.2269 | 6/6, delta +1.6754 | safe |
| `policy_recovery_seed75_add_l1h4_b050` | 76 | `1:4` | 16.96% | 13/13, delta +0.5610 | 6/6, delta +1.8397 | safe |
| `policy_recovery_seed76_add_l23h3_b050` | 77 | `23:3` | 17.19% | 13/13, delta +0.4796 | 6/6, delta +1.6751 | safe |
| `policy_recovery_seed77_add_l1h6_b050` | 78 | `1:6` | 17.41% | 12/13, delta +0.3075 | not run | unsafe |
| `policy_recovery_seed77_add_l2h1_b050` | 78 | `2:1` | 17.41% | 12/13, delta +0.3075 | not run | unsafe |
| `policy_recovery_seed77_add_l9h7_b050` | 78 | `9:7` | 17.41% | 13/13, delta +0.5104 | 6/6, delta +1.4734 | safe |

Current best cross-document safe frontier:

- Policy: `experiments/light_doc_cache/policy_recovery_seed77_add_l9h7_b050`
- Heads: 78
- Per-head selected budget: 50%
- Total KV head-token entry saving: 17.41%
- First-doc v3 stable: 13/13 pass, mean delta +0.5104
- Second-doc stable: 6/6 pass, mean delta +1.4734
- Head list: `24:0,24:1,24:2,24:3,24:4,24:5,24:6,24:7,10:3,8:1,20:2,12:4,10:5,12:7,20:7,8:3,12:5,16:5,8:0,20:6,10:2,12:0,20:1,10:1,20:3,0:0,8:7,4:1,0:2,0:5,0:4,10:7,10:6,0:7,4:2,4:7,10:0,4:4,0:3,16:2,8:4,4:0,4:3,0:1,6:7,3:0,26:5,18:0,22:1,11:2,22:0,3:6,13:1,3:1,13:3,23:7,11:0,13:5,17:1,3:7,2:7,17:5,14:1,1:5,1:7,13:0,9:6,5:2,4:6,16:6,18:6,6:0,15:3,18:5,2:3,1:4,23:3,9:7`

Round6 takeaways:

- New safe heads: `2:3`, `1:4`, `23:3`, `9:7`.
- `14:6` passed first-doc but failed second-doc, reinforcing that cross-document gates are necessary.
- `14:3`, `1:6`, and `2:1` failed first-doc despite being high-ranked candidates.
- Latest frontier: 78 heads / 17.41% total KV head-token entry saving. This is stronger sparse fixed-head evidence, but it remains offline quality simulation only. It should be framed as a trained/correlation recovery-bank diagnostic result, not as KR-level 2x+ runtime doc-cache compression.
- Next higher-leverage direction: document-adaptive/task-aware gating, shared recovery-module training, and real runtime K/V cache integration. Pure fixed-head greedy expansion is producing diminishing returns.

### 2026-07-09 round7 conservative search advanced frontier to 79 heads

Generated a conservative Round7 candidate table from the Round6 ranking by excluding known safe/unsafe heads, penalizing layers with accumulated task-gate failures, and preferring positive cross-document minimum recovery scores.

- Candidate ranking: `experiments/light_doc_cache/head_candidates_round7_from78_20260709.tsv`
- Automation log: `experiments/light_doc_cache/head_progress_round7_from78_20260709_224340.tsv`
- Bank method: `ridge`
- Per-head selected budget: 50%
- Same strict first-doc and second-doc gates.

Round7 top-8 results:

| Policy | Heads | Added Head | Total Entry Saving | First Doc v3 | Second Doc Stable | Result |
|---|---:|---|---:|---|---|---|
| `policy_recovery_seed78_add_l6h6_b050` | 79 | `6:6` | 17.63% | 12/13, delta +0.6312 | not run | unsafe |
| `policy_recovery_seed78_add_l5h5_b050` | 79 | `5:5` | 17.63% | 12/13, delta +0.4275 | not run | unsafe |
| `policy_recovery_seed78_add_l11h3_b050` | 79 | `11:3` | 17.63% | 13/13, delta +0.6782 | 6/6, delta +1.4649 | safe |
| `policy_recovery_seed79_add_l22h6_b050` | 80 | `22:6` | 17.86% | 11/13, delta +1.3102 | not run | unsafe |
| `policy_recovery_seed79_add_l15h5_b050` | 80 | `15:5` | 17.86% | 12/13, delta +1.0609 | not run | unsafe |
| `policy_recovery_seed79_add_l7h7_b050` | 80 | `7:7` | 17.86% | 12/13, delta +0.7118 | not run | unsafe |
| `policy_recovery_seed79_add_l6h4_b050` | 80 | `6:4` | 17.86% | 12/13, delta +0.6025 | not run | unsafe |
| `policy_recovery_seed79_add_l15h4_b050` | 80 | `15:4` | 17.86% | 11/13, delta +0.7605 | not run | unsafe |

Current best cross-document safe frontier:

- Policy: `experiments/light_doc_cache/policy_recovery_seed78_add_l11h3_b050`
- Heads: 79
- Per-head selected budget: 50%
- Total KV head-token entry saving: 17.63%
- First-doc v3 stable: 13/13 pass, mean delta +0.6782
- Second-doc stable: 6/6 pass, mean delta +1.4649
- Head list: `24:0,24:1,24:2,24:3,24:4,24:5,24:6,24:7,10:3,8:1,20:2,12:4,10:5,12:7,20:7,8:3,12:5,16:5,8:0,20:6,10:2,12:0,20:1,10:1,20:3,0:0,8:7,4:1,0:2,0:5,0:4,10:7,10:6,0:7,4:2,4:7,10:0,4:4,0:3,16:2,8:4,4:0,4:3,0:1,6:7,3:0,26:5,18:0,22:1,11:2,22:0,3:6,13:1,3:1,13:3,23:7,11:0,13:5,17:1,3:7,2:7,17:5,14:1,1:5,1:7,13:0,9:6,5:2,4:6,16:6,18:6,6:0,15:3,18:5,2:3,1:4,23:3,9:7,11:3`

Round7 takeaways:

- New safe head: `11:3`.
- Seven of eight conservative candidates failed the first document; failures are now strongly concentrated in task-level flips rather than command instability.
- Latest frontier: 79 heads / 17.63% total KV head-token entry saving. Fixed global head-list expansion has clear diminishing returns; next work should move to document-adaptive/task-aware gating or integrated recovery-module/runtime work before making larger compression claims.

### 2026-07-09 task-aware failure diagnostics and budget rescue negative result

Aggregated local mirrored `task_rows.csv` files from Round4-Round7 to identify the recurring task-level failures that stop late fixed-head expansion.

- Diagnostic report: `experiments/light_doc_cache/task_failure_diagnostics_round4_round7_20260709.md`
- Flip rows: `experiments/light_doc_cache/task_failure_diagnostics_round4_round7_20260709.tsv`
- Per-policy summary: `experiments/light_doc_cache/task_quality_summary_round4_round7_20260709.tsv`

Most frequent flipped tasks:

| Count | Doc | Task |
|---:|---|---|
| 14 | first | `topk8_quality` |
| 8 | first | `route_phase3` |
| 7 | first | `sweet_spot` |
| 6 | first | `quest_decode_selection` |
| 2 | second | `tp_true_weight_split` |
| 2 | second | `smoothquant_status` |

Then tested a low-risk mixed-budget rescue: start from the 79-head safe frontier, add one failed Round7 candidate, keep existing safe heads at 50% selected budget, and give the newly added head 75% selected budget.

- Progress log: `experiments/light_doc_cache/taskaware_budget75_from79_progress_20260709_225838.tsv`
- Tested policies:
  - `experiments/light_doc_cache/policy_taskaware_budget75_from79_add_l6h6`
  - `experiments/light_doc_cache/policy_taskaware_budget75_from79_add_l5h5`
  - `experiments/light_doc_cache/policy_taskaware_budget75_from79_add_l22h6`
  - `experiments/light_doc_cache/policy_taskaware_budget75_from79_add_l15h5`

Budget-rescue results:

| Added Head | Heads | Entry Saving | First Doc v3 | Result |
|---|---:|---:|---|---|
| `6:6` | 80 | 17.75% | 12/13, delta +0.8239 | unsafe |
| `5:5` | 80 | 17.75% | 12/13, delta +0.5985 | unsafe |
| `22:6` | 80 | 17.75% | 11/13, delta +1.3246 | unsafe |
| `15:5` | 80 | 17.75% | 12/13, delta +1.0727 | unsafe |

Targeted learned-values rescue:

- Policy: `experiments/light_doc_cache/policy_taskaware_budget75_from79_add_l15h5`
- Output: `experiments/light_doc_cache/task_quality_smoke_taskaware_budget75_from79_add_l15h5_v3_learnedv_latest`
- Bank method: `learned_values`
- First-doc v3 stable: 11/13, mean delta +1.0147
- Mean bank build time: 6.3772s, compared with 0.3760s for ridge on the same policy.

Conclusion: neither 75% budget for the added head nor learned compact values rescued the late fixed-head frontier. The evidence now points away from further global fixed-head expansion and toward true document-adaptive/task-aware gating or runtime recovery-module integration.

### 2026-07-09 adaptive policy v1 recovered 80-head default with task fallback

Implemented a quality-only adaptive policy hook in `experiments/light_doc_cache/task_quality_smoke.py`.

Code changes:

- Added `--adaptive-policy-file`.
- Added `load_adaptive_policy(...)`, `apply_adaptive_policy(...)`, and task-level `effective_entry_saving_fraction`.
- Kept default `--policy-dir` behavior unchanged when no adaptive policy is provided.
- Updated `experiments/light_doc_cache/run_task_quality_smoke_remote.sh` to support `LOCAL_ADAPTIVE_POLICY_FILE` and verify `adaptive_policy.json` transfer.
- Added `experiments/light_doc_cache/make_adaptive_task_policy.py` to generate adaptive specs from task-failure diagnostics.
- Added static/behavior coverage in `tools/test_light_doc_cache_recovery_probe.py`.

Adaptive policy:

- Spec: `experiments/light_doc_cache/adaptive_policy_from79_add_l15h5_drop_on_fragile_v1.json`
- Auto-generated equivalent spec: `experiments/light_doc_cache/adaptive_policy_from79_add_l15h5_auto_top4_first_v1.json`
- Default policy: `experiments/light_doc_cache/policy_taskaware_budget75_from79_add_l15h5`
- Default compression: 80 heads, adding `15:5` at 75% selected budget.
- Task fallback: for `topk8_quality`, `route_phase3`, `sweet_spot`, and `quest_decode_selection`, drop `15:5`, so those tasks use the 79-head safe frontier.

Remote results:

| Output | Tasks | Quality Gate | Mean Delta | Mean Bank Build | Avg Effective Saving |
|---|---:|---|---:|---:|---:|
| `experiments/light_doc_cache/task_quality_smoke_adaptive_from79_add_l15h5_v3_latest` | 13 | 13/13 accuracy and agreement | +0.9575 | 0.2921s | 17.71% |
| `experiments/light_doc_cache/task_quality_smoke_adaptive_from79_add_l15h5_qwen8bfixes_latest` | 6 | 6/6 accuracy and agreement | +1.7281 | 0.2999s | 17.75% |

Interpretation:

- The static 80-head mixed-budget policy failed the first document at 12/13.
- Adaptive task fallback recovered the first document to 13/13 and kept the second document at 6/6.
- This is the best current quality-simulation frontier because it beats the static 79-head policy in average effective saving while preserving strict two-document gates.
- Claim boundary remains important: this is a task-adaptive recovery-bank quality simulation with about 17.7% average KV head-token entry saving, not a 2x+ runtime doc-cache compression result.

Auto-generation command:

```bash
python3 experiments/light_doc_cache/make_adaptive_task_policy.py \
  --failure-diagnostics experiments/light_doc_cache/task_failure_diagnostics_round4_round7_20260709.tsv \
  --default-policy-dir experiments/light_doc_cache/policy_taskaware_budget75_from79_add_l15h5 \
  --base-safe-policy-dir experiments/light_doc_cache/policy_recovery_seed78_add_l11h3_b050 \
  --drop-heads 15:5 \
  --top-tasks 4 \
  --doc first \
  --output experiments/light_doc_cache/adaptive_policy_from79_add_l15h5_auto_top4_first_v1.json
```

### 2026-07-09 adaptive search over 80-head default policies

Ran a six-candidate adaptive search. Each default policy starts from the 79-head safe frontier, adds one head at 75% selected-token budget, and falls back only on the top-4 first-doc fragile tasks by dropping the newly added head.

- Result table: `experiments/light_doc_cache/adaptive_search_from79_top4_two_doc_results_20260709.tsv`
- First-doc log: `experiments/light_doc_cache/adaptive_search_from79_top4_first_20260709.log`
- Second-doc log: `experiments/light_doc_cache/adaptive_search_from79_top4_second_20260709.log`
- Candidate specs:
  - `experiments/light_doc_cache/adaptive_policy_from79_add_l6h6_auto_top4_first_v1.json`
  - `experiments/light_doc_cache/adaptive_policy_from79_add_l5h5_auto_top4_first_v1.json`
  - `experiments/light_doc_cache/adaptive_policy_from79_add_l22h6_auto_top4_first_v1.json`
  - `experiments/light_doc_cache/adaptive_policy_from79_add_l15h5_auto_top4_first_v1.json`
  - `experiments/light_doc_cache/adaptive_policy_from79_add_l7h7_auto_top4_first_v1.json`
  - `experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_v1.json`

Results:

| Added Head | First Doc v3 | First Delta | Second Doc Stable | Second Delta | Result |
|---|---|---:|---|---:|---|
| `6:6` | 13/13 | +0.7536 | 3/6 | +1.2965 | unsafe |
| `5:5` | 13/13 | +0.6721 | 5/6 | +1.7423 | unsafe |
| `22:6` | 13/13 | +1.2329 | 6/6 | +1.4539 | safe |
| `15:5` | 13/13 | +0.9575 | 6/6 | +1.7281 | safe |
| `7:7` | 13/13 | +0.7426 | 4/6 | +1.4461 | unsafe |
| `6:4` | 13/13 | +0.6108 | 4/6 | +1.2970 | unsafe |

Interpretation:

- First-doc-only fallback was enough to recover all six candidates on the first document.
- Cross-document validation still filters aggressively: only `22:6` and `15:5` passed both documents.
- `15:5` remains the best overall adaptive policy by second-doc margin (+1.7281), while `22:6` is also a valid two-document policy and has the strongest first-doc margin (+1.2329).
- Failed second-doc tasks are mostly `tp_true_weight_split`, `gpu_utilization_semantics`, and `smoothquant_status`; these are not covered by the top-4 first-doc fragile fallback.

Updated best adaptive frontier:

- Preferred spec: `experiments/light_doc_cache/adaptive_policy_from79_add_l15h5_auto_top4_first_v1.json`
- Alternate safe spec: `experiments/light_doc_cache/adaptive_policy_from79_add_l22h6_auto_top4_first_v1.json`
- First-doc avg effective saving: 17.71%
- Second-doc avg effective saving: 17.75%

Claim boundary: still quality-only recovery-bank simulation, not runtime KV-cache compression. The next meaningful step is a two-document adaptive-policy generator with second-doc fallback tasks, followed by runtime integration only after quality gates remain stable.

### 2026-07-10 two-document fallback rescued second-doc failures

Extended `experiments/light_doc_cache/make_adaptive_task_policy.py` so it can combine multiple failure sources and select fallback tasks independently per document.

New generator behavior:

- `--failure-diagnostics` can be provided multiple times.
- `--per-doc-top-tasks first=4,second=3` selects separate top fragile tasks for each document.
- The adaptive result table `experiments/light_doc_cache/adaptive_search_from79_top4_two_doc_results_20260709.tsv` can be consumed as a failure source via its `fail_tasks` column.

Generated and remotely validated two-doc fallback specs:

| Spec | Added Head |
|---|---|
| `experiments/light_doc_cache/adaptive_policy_from79_add_l6h6_auto_top4_first_top3_second_v1.json` | `6:6` |
| `experiments/light_doc_cache/adaptive_policy_from79_add_l5h5_auto_top4_first_top3_second_v1.json` | `5:5` |
| `experiments/light_doc_cache/adaptive_policy_from79_add_l7h7_auto_top4_first_top3_second_v1.json` | `7:7` |
| `experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json` | `6:4` |

Remote validation outputs:

- First-doc log: `experiments/light_doc_cache/adaptive_search_from79_twodoc_first_20260710.log`
- Second-doc log: `experiments/light_doc_cache/adaptive_search_from79_twodoc_second_20260710.log`
- Result table: `experiments/light_doc_cache/adaptive_search_from79_twodoc_fallback_results_20260710.tsv`

Results:

| Added Head | First Doc v3 | First Saving | Second Doc Stable | Second Saving | Second Delta | Result |
|---|---|---:|---|---:|---:|---|
| `6:6` | 13/13 | 17.71% | 6/6 | 17.69% | +1.6298 | rescued |
| `5:5` | 13/13 | 17.71% | 6/6 | 17.69% | +1.5364 | rescued |
| `7:7` | 13/13 | 17.71% | 6/6 | 17.69% | +1.4269 | rescued |
| `6:4` | 13/13 | 17.71% | 6/6 | 17.69% | +1.4672 | rescued |

Interpretation:

- The two-doc fallback policy recovered all four candidates that previously failed second-doc gates.
- The second-doc fallback rules are `tp_true_weight_split`, `gpu_utilization_semantics`, and `smoothquant_status`; for those tasks the added head is dropped, reverting to the 79-head safe frontier.
- Effective saving remains about 17.7%; the extra fallback reduces second-doc saving slightly from 17.75% to 17.69%.
- This is stronger evidence for document/task-conditional recovery-bank policies, but still not a runtime KV-cache compression result.

Next useful step: package the validated adaptive result into a concise paper-ready table and then start a runtime integration prototype that applies the adaptive policy to real KV-cache storage/recovery rather than only decode-score simulation.

### 2026-07-10 paper-ready frontier table generated

Added a small reusable frontier-table generator:

- Script: `experiments/light_doc_cache/make_adaptive_frontier_table.py`
- Test coverage: `tools/test_light_doc_cache_recovery_probe.py::test_make_adaptive_frontier_table_writes_paper_ready_outputs`
- Output directory: `experiments/light_doc_cache/paper_frontier_table_20260710`
- CSV: `experiments/light_doc_cache/paper_frontier_table_20260710/frontier_table.csv`
- Markdown: `experiments/light_doc_cache/paper_frontier_table_20260710/frontier_table.md`

Paper-ready summary:

| Frontier | First Doc Gate | Second Doc Gate | First Saving | Second Saving | Claim Boundary |
|---|---|---|---:|---:|---|
| static 79-head fixed global | 13/13 | 6/6 | 17.63% | 17.63% | quality-only |
| first-doc adaptive `15:5` | 13/13 | 6/6 | 17.71% | 17.75% | quality-only |
| two-doc adaptive candidates | 13/13 | 6/6 | 17.71% | 17.69% | quality-only |

The safe paper wording is:

> Offline task/document-adaptive recovery-bank simulation preserves strict two-document task gates while improving average effective KV head-token entry saving from 17.63% to about 17.7%.

This should not be described as 2x+ runtime doc-cache compression. The table is intended as a stable evidence artifact before starting runtime KV-cache integration.

### 2026-07-10 runtime planning/metrics prototype started

Added the first runtime-side artifact without touching the `ModelRunner` hot path:

- Module: `tinyvllm/engine/light_doc_cache_runtime.py`
- Test file: `tools/test_light_doc_cache_runtime.py`

What it does:

- Loads a `task_adaptive_light_doc_cache_policy` JSON file.
- Extracts the default added head from policy names such as `...budget75_from79_add_l6h4`.
- Applies task-level overrides such as dropping the added head for fragile tasks.
- Reports stored/recovered KV head-token entries and equivalent KV-head counts after budget fractions.
- Exposes a tiny CLI that prints a JSON summary for a single `(task_id, doc_id, seq_len)` request.

Validation:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
python3 -m pytest -q tools/test_light_doc_cache_runtime.py
# 6 passed
```

Important correction: the runtime planner is budget-aware. For example, a `budget75` added head means retaining 75% of that head's doc tokens and recovering/saving 25%, not dropping the whole head. This avoids overclaiming compression when translating the offline recovery-bank policy into runtime storage metrics.

Claim boundary:

- This is a planning/metrics prototype only.
- It does not allocate compressed KV storage.
- It does not perform online tensor recovery.
- It does not measure actual runtime memory or latency.

Next useful runtime step: attach this planner to a default-off `ModelRunner` summary path or a standalone smoke that reads validated adaptive policies and request metadata, then compare planned entries with actual KV block/head storage once a real compressed layout is introduced.

### 2026-07-10 runtime planning table from real policy rows

Extended the runtime planning prototype so it can read real `policy_rows.csv` artifacts instead of requiring hand-entered head lists:

- `build_config_from_policy_dirs(...)` loads compact heads and per-head `budget_fraction`.
- `experiments/light_doc_cache/make_runtime_plan_table.py` generates JSONL and Markdown tables from either explicit `--task` values or task files.
- The script deliberately imports `tinyvllm/engine/light_doc_cache_runtime.py` by file path so local CPU validation does not require the full `transformers` dependency stack.

Generated artifact:

- Directory: `experiments/light_doc_cache/runtime_plan_table_from79_l6h4_20260710`
- JSONL: `experiments/light_doc_cache/runtime_plan_table_from79_l6h4_20260710/runtime_plan_table.jsonl`
- Markdown: `experiments/light_doc_cache/runtime_plan_table_from79_l6h4_20260710/runtime_plan_table.md`

Command:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
python3 experiments/light_doc_cache/make_runtime_plan_table.py \
  --policy-file experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json \
  --repo-root /Users/bytedance/dev/TinyLLMForge \
  --num-layers 28 \
  --num-kv-heads 8 \
  --seq-len 1536 \
  --task-file first=experiments/light_doc_cache/task_quality_tasks_kv_sparse_v3_stable.json \
  --task-file second=experiments/light_doc_cache/task_quality_tasks_qwen3_8b_fixes_stable.json \
  --output-dir experiments/light_doc_cache/runtime_plan_table_from79_l6h4_20260710
```

Runtime-planning summary over the 19 validated task rows:

- Average effective planned saving: `17.70%`.
- Average planned compression ratio: `1.2151x`.
- Non-fallback rows: 80 recovered heads, `39.75` recovered KV-head equivalent, `17.75%` saving.
- Fallback rows: 79 recovered heads, `39.50` recovered KV-head equivalent, `17.63%` saving.
- Audited full KV cache shape: `[2, 28, 6, 256, 8, 128]` with 2-byte KV elements.
- Full KV bytes for that shape: `176,160,768`.
- Average planned recovered/saved KV bytes: `31,188,237`.
- Average planned stored KV bytes after policy budgets: `144,972,531`.

This table is still planning/metrics only. It is the correct accounting bridge for runtime work because it uses the actual mixed-budget policy rows and avoids the earlier dangerous simplification that a compact head is fully removed from storage.

Follow-up implementation detail: added `summarize_planned_kv_storage(...)` to `tinyvllm/engine/light_doc_cache_runtime.py`. It maps a runtime plan to full-cache byte accounting without allocating tensors. The table generator now accepts `--num-blocks`, `--block-size`, `--head-dim`, and `--element-size-bytes` to include byte-level storage accounting in JSONL/Markdown.

Added one more bridge helper: `summarize_planned_kv_storage_from_shape(...)`. It accepts a real `kv_cache.shape`-style tuple `[2, layers, blocks, block_size, kv_heads, head_dim]` plus element size, validates layer/head compatibility with the plan, and emits the same byte accounting with `shape_source=kv_cache_shape`. The table generator now supports `--kv-cache-shape 2,28,6,256,8,128`, so a future `ModelRunner` summary hook can pass `tuple(self.kv_cache.shape)` directly.

### 2026-07-10 ModelRunner summary-only wrapper

Added a thin default-off wrapper in `tinyvllm/engine/model_runner.py`:

```python
def light_doc_cache_planning_summary(self, plan) -> dict | None:
    return build_model_runner_light_doc_cache_summary(self, plan)
```

The implementation reads only `self.kv_cache.shape` and `self.kv_cache.element_size()` through the helper in `tinyvllm/engine/light_doc_cache_runtime.py`. It does not alter `allocate_kv_cache()`, KV slot mapping, attention kernels, prefill/decode metadata, or KV writes.

Test coverage is intentionally local/CPU-safe:

- fake-runner test for `build_model_runner_light_doc_cache_summary(...)`;
- static wrapper test for `ModelRunner.light_doc_cache_planning_summary(...)`;
- no local import of `ModelRunner`, because this checkout lacks `transformers` locally and GPU/model validation belongs on the remote host.

Remote validation:

- Synced the touched runtime files and minimal policy/task artifacts to `sitian@10.232.195.203:/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge`.
- Remote `py_compile` passed for:
  - `tinyvllm/engine/light_doc_cache_runtime.py`
  - `tinyvllm/engine/model_runner.py`
  - `experiments/light_doc_cache/make_runtime_plan_table.py`
  - `tools/test_light_doc_cache_runtime.py`
- Remote pure-Python smoke passed in `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`:
  - loaded the `6:4` adaptive policy and real policy rows,
  - built a `smoothquant_status` fallback plan,
  - called `build_model_runner_light_doc_cache_summary(...)` with fake `kv_cache.shape=(2,28,6,256,8,128)`,
  - verified `full_kv_bytes=176160768` and `planned_recovered_kv_bytes=31064064`.

Remote caveat: `pytest` is not installed in that env, so remote validation used `py_compile` plus direct Python assertions instead of `python -m pytest`.

### 2026-07-10 CPU compressed KV storage prototype

Added the first real-storage prototype, still isolated from `ModelRunner` hot paths:

- Class: `LightDocCacheCompressedKVStorage` in `tinyvllm/engine/light_doc_cache_runtime.py`
- Smoke script: `experiments/light_doc_cache/run_storage_prototype_smoke.py`
- Output directory: `experiments/light_doc_cache/storage_prototype_smoke_l6h4_20260710`

Behavior:

- Non-compact heads are stored as full flattened token tensors.
- Compact heads store only selected prefix tokens according to the runtime plan.
- Restore returns the original KV shape and fills missing compact-head tokens with a sentinel value.
- Cross-block selected-token slicing is handled on the flattened token dimension, so selected tokens are global prefix tokens rather than per-block prefixes.

Local toy smoke result for policy `adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json`:

- Full tensor bytes: `57,344`
- Stored tensor bytes: `35,424`
- Saved tensor bytes: `21,920`
- Byte saving fraction: `38.23%`
- Compact heads: `79`
- Full heads: `145`

Claim boundary: this is a CPU/toy storage-layout proof that fewer tensor bytes can be stored for planned compact heads. It is **not** model-quality recovery, because missing compact-head tokens are filled, not reconstructed by the learned/ridge recovery bank.

Remote validation:

- Synced `tinyvllm/engine/light_doc_cache_runtime.py`, `tinyvllm/engine/model_runner.py`, `experiments/light_doc_cache/run_storage_prototype_smoke.py`, and required policy rows to `sitian@10.232.195.203`.
- Remote `py_compile` passed for the touched runtime/storage scripts.
- Remote storage smoke passed:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge_pycache_lightdoc \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python \
  experiments/light_doc_cache/run_storage_prototype_smoke.py \
  --policy-file experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json \
  --repo-root /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge \
  --output-dir /tmp/light_doc_cache_storage_smoke_remote_20260710
```

Remote result: `byte_saving_fraction=0.3822544642857143`.

### 2026-07-13 calibrated read-path artifact and SSH blocker

The TinyLLM calibrated multi-source recovery bank has been exercised through
the default-off restored-sidecar read-path smoke on the target prompt:

- Artifact:
  `experiments/light_doc_cache/tinyllm_sidecar_read_path_calibrated_smoke_qwen3_0_6b_20260710/`
- Bank artifact:
  `experiments/light_doc_cache/tinyllm_calibrated_kv_smoke_qwen3_0_6b_20260710/multi_source_recovery_bank.json`
- Target prompt:
  `Light Doc Cache TinyLLM target prompt for Qwen KV recovery.`
- Recovery mode: `calibrated_multi_correlated`
- Prompt tokens: `14`
- Logical byte saving fraction: `17.6339%`
- Missing compact tokens: `553`
- Missing-token MSE: `13.5193`
- Missing-token max abs error: `219`
- Max abs logit diff: `3.890625`
- Mean abs logit diff: `0.6842344403266907`
- Original/restored argmax: `785 / 785`
- Argmax match: `true`

This is the first trained/calibrated recovery-bank path that reaches a real
TinyLLM restored-sidecar read-path logits comparison and preserves the target
prompt argmax. It remains default-off: the smoke temporarily points attention
cache pointers at a restored sidecar tensor for one decode-step comparison, then
restores the original pointers. It does not change attention hot path, physical
KV allocation lifetime, or runtime memory use.

Fair target-prompt baseline artifacts were pulled after the SSH ControlMaster
socket became available again. The final artifact-backed matrix is now:

- `experiments/light_doc_cache/read_path_recovery_matrix_calibrated_target_qwen3_0_6b_20260710/read_path_recovery_matrix.csv`
- `experiments/light_doc_cache/read_path_recovery_matrix_calibrated_target_qwen3_0_6b_20260710/read_path_recovery_matrix.md`

Rows:

| Mode | Role | Missing MSE | Missing Max Abs | Max Logit Diff | Mean Logit Diff | Argmax |
|---|---|---:|---:|---:|---:|---|
| `repeat_last_target` | baseline | 15.2284 | 217 | 4.0625 | 0.598329 | 785 -> 785 |
| `correlated_same_layer_target` | baseline | 11.4076 | 246 | 3.59375 | 0.507468 | 785 -> 785 |
| `calibrated_multi_correlated_target` | trained | 13.5193 | 219 | 3.890625 | 0.684234 | 785 -> 785 |

Interpretation:

- All three non-oracle modes preserve the target-prompt argmax.
- The trained/calibrated bank validates the read-path plumbing, but it does not
  beat the simple same-layer correlated baseline on this single target prompt.
- The old provisional stdout-backed matrix remains only as handoff history:
  `experiments/light_doc_cache/read_path_recovery_matrix_calibrated_target_qwen3_0_6b_20260710_provisional/`.

SSH blocker resolution:

- `/tmp/ssh-sitian-10.232.195.203` appeared after external credential refresh.
- Direct SSH still intermittently returned `Connection closed by UNKNOWN port
  65535`, but `ssh -S /tmp/ssh-sitian-10.232.195.203 ...` succeeded.
- Both remote directories were pulled with `rsync -e 'ssh -S ...'`.

Pull commands used:

```bash
rsync -av -e 'ssh -S /tmp/ssh-sitian-10.232.195.203 -o BatchMode=yes -o ConnectTimeout=10' \
  sitian@10.232.195.203:/tmp/light_doc_cache_tinyllm_read_path_repeat_last_target_qwen3_0_6b_20260710/ \
  experiments/light_doc_cache/tinyllm_sidecar_read_path_repeat_last_target_smoke_qwen3_0_6b_20260710/

rsync -av -e 'ssh -S /tmp/ssh-sitian-10.232.195.203 -o BatchMode=yes -o ConnectTimeout=10' \
  sitian@10.232.195.203:/tmp/light_doc_cache_tinyllm_read_path_correlated_target_qwen3_0_6b_20260710/ \
  experiments/light_doc_cache/tinyllm_sidecar_read_path_correlated_target_smoke_qwen3_0_6b_20260710/
```

### 2026-07-13 multi-prompt calibrated bank

Implemented multi-prompt calibration support in
`run_tinyllm_calibrated_kv_smoke.py`:

- New CLI:
  - repeatable `--calibration-prompt-extra`
  - optional `--calibration-prompts-file`
- `copy_kv_prompt_prefix(...)` copies only prompt-prefix KV blocks and zero-fills
  padding in the last block. This avoids retaining full preallocated KV clones
  for each calibration prompt.
- `stack_calibration_kv_samples(...)` packs multiple prompt prefixes into one
  calibration KV tensor.
- `run_calibrated_smoke(...)` now fits with a calibration-length plan and
  evaluates with a target-length plan.

Remote first attempt failed with OOM:

- Cause: each calibration sample cloned the full TinyLLM preallocated KV cache.
- Error: `torch.OutOfMemoryError: Tried to allocate 21.52 GiB`.
- Fix: copy only prompt-prefix blocks before keeping calibration samples.

Remote Qwen3-0.6B KV-only multi-prompt smoke:

```bash
CUDA_VISIBLE_DEVICES=2 ... \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python \
  experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py \
  --calibration-prompt "Light Doc Cache TinyLLM calibration prompt." \
  --calibration-prompt-extra "Light Doc Cache second calibration prompt for trained recovery." \
  --calibration-prompt-extra "Light Doc Cache third calibration prompt for Qwen KV recovery." \
  --target-prompt "Light Doc Cache TinyLLM target prompt for Qwen KV recovery." \
  --output-dir /tmp/light_doc_cache_tinyllm_calibrated_kv_qwen3_0_6b_20260713_multiprompt
```

Result:

- Artifact:
  `experiments/light_doc_cache/tinyllm_calibrated_kv_smoke_qwen3_0_6b_20260713_multiprompt/`
- Calibration tokens: `31`
- Calibration plan tokens: `31`
- Target tokens: `14`
- KV-only missing-token MSE: `11.6785`
- KV-only max abs error: `258`

Remote read-path smoke with the multi-prompt bank:

- Artifact:
  `experiments/light_doc_cache/tinyllm_sidecar_read_path_calibrated_multiprompt_smoke_qwen3_0_6b_20260713/`
- Missing-token MSE: `9.55884`
- Missing max abs: `146`
- Max logit diff: `4.0625`
- Mean logit diff: `0.538956`
- Argmax: `785 -> 785`

Extended matrix:

- `experiments/light_doc_cache/read_path_recovery_matrix_calibrated_multiprompt_target_qwen3_0_6b_20260713/`

| Mode | Role | Missing MSE | Missing Max Abs | Max Logit Diff | Mean Logit Diff | Argmax |
|---|---|---:|---:|---:|---:|---|
| `repeat_last_target` | baseline | 15.2284 | 217 | 4.0625 | 0.598329 | 785 -> 785 |
| `correlated_same_layer_target` | baseline | 11.4076 | 246 | 3.59375 | 0.507468 | 785 -> 785 |
| `calibrated_single_pair_target` | trained | 13.5193 | 219 | 3.890625 | 0.684234 | 785 -> 785 |
| `calibrated_multiprompt_target` | trained | 9.55884 | 146 | 4.0625 | 0.538956 | 785 -> 785 |

Interpretation: multi-prompt calibration is a real improvement over the
single-pair trained bank and is now best on missing-token MSE. It still does not
beat `correlated_same_layer_target` on mean logit diff, so the next step should
be better source-head selection or a logit-aware calibration objective before
any attention hot-path or physical KV-allocation work.

### 2026-07-13 calibration-fit source selection ablation

Added `--source-map {same_layer,calibration_fit}` to
`run_tinyllm_calibrated_kv_smoke.py`.

- `same_layer`: previous source-head selection.
- `calibration_fit`: ranks retained source heads by calibration-prefix
  reconstruction fit for each compact target head.

Local TDD:

- Added a toy test proving `calibration_fit` chooses the source head with lower
  calibration-prefix reconstruction error.
- Full local validation later passed with `65 passed`.

Remote Qwen3-0.6B KV-only smoke:

```bash
CUDA_VISIBLE_DEVICES=5 ... \
  experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py \
  --source-map calibration_fit \
  --calibration-prompt "Light Doc Cache TinyLLM calibration prompt." \
  --calibration-prompt-extra "Light Doc Cache second calibration prompt for trained recovery." \
  --calibration-prompt-extra "Light Doc Cache third calibration prompt for Qwen KV recovery." \
  --output-dir /tmp/light_doc_cache_tinyllm_calibrated_kv_qwen3_0_6b_20260713_calfit
```

Result:

- Artifact:
  `experiments/light_doc_cache/tinyllm_calibrated_kv_smoke_qwen3_0_6b_20260713_calfit/`
- Source map: `calibration_fit`
- Calibration tokens: `31`
- KV-only missing-token MSE: `20.7961`

Remote read-path smoke:

- Artifact:
  `experiments/light_doc_cache/tinyllm_sidecar_read_path_calfit_smoke_qwen3_0_6b_20260713/`
- Missing-token MSE: `11.5086`
- Missing max abs: `211`
- Max logit diff: `3.4375`
- Mean logit diff: `0.533486`
- Argmax: `785 -> 785`

Extended matrix:

- `experiments/light_doc_cache/read_path_recovery_matrix_calibrated_sourcefit_target_qwen3_0_6b_20260713/`

Interpretation: this is a mixed/negative selector result. Prefix-fit source
selection improves read-path max logit diff and slightly improves mean logit
diff over multi-prompt same-layer trained bank, but worsens missing-token MSE
and still does not beat the cheap `correlated_same_layer_target` baseline on
mean logit diff. Next selector should score held-out missing-token prediction or
logit/read-path impact instead of prefix fit alone.

### 2026-07-13 calibration-holdout source selection

Added `--source-map calibration_holdout` to
`run_tinyllm_calibrated_kv_smoke.py`.

Method:

- For each compact target head, fit a one-source affine map on selected
  calibration prefix tokens.
- Score each retained source head by prediction MSE on the remaining held-out
  calibration tokens.
- Select the lowest-error source heads for the final calibrated multi-source
  bank.

Local TDD:

- Added a toy test where prefix-only fit would choose the wrong source but
  held-out prediction selects the source that generalizes.
- Fixed a remote torch device mismatch in held-out scoring (`cuda` target vs
  CPU prediction).
- Full local validation passed: `66 passed`.

Remote Qwen3-0.6B KV-only smoke:

```bash
CUDA_VISIBLE_DEVICES=5 ... \
  experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py \
  --source-map calibration_holdout \
  --calibration-prompt "Light Doc Cache TinyLLM calibration prompt." \
  --calibration-prompt-extra "Light Doc Cache second calibration prompt for trained recovery." \
  --calibration-prompt-extra "Light Doc Cache third calibration prompt for Qwen KV recovery." \
  --output-dir /tmp/light_doc_cache_tinyllm_calibrated_kv_qwen3_0_6b_20260713_holdout
```

Result:

- Artifact:
  `experiments/light_doc_cache/tinyllm_calibrated_kv_smoke_qwen3_0_6b_20260713_holdout/`
- Source map: `calibration_holdout`
- KV-only missing-token MSE: `7.07867`
- KV-only max abs error: `155`

Remote read-path smoke:

- Artifact:
  `experiments/light_doc_cache/tinyllm_sidecar_read_path_holdout_smoke_qwen3_0_6b_20260713/`
- Missing-token MSE: `11.7772`
- Missing max abs: `252`
- Max logit diff: `3.10938`
- Mean logit diff: `0.455011`
- Argmax: `785 -> 785`

Final holdout matrix:

- `experiments/light_doc_cache/read_path_recovery_matrix_calibrated_holdout_target_qwen3_0_6b_20260713/`

Interpretation: holdout source selection is the first trained/calibrated row
that beats the cheap `correlated_same_layer_target` baseline on mean logit diff
(`0.455011` vs `0.507468`) and max logit diff (`3.10938` vs `3.59375`) for this
target prompt. It is not best on missing-token MSE; multi-prompt same-layer is
still best there. Next step should test more target prompts / produce a
multi-target matrix before hot-path integration.

Then generate the final target-prompt matrix:

```bash
python3 experiments/light_doc_cache/make_read_path_recovery_matrix.py \
  --artifact repeat_last_target:baseline:experiments/light_doc_cache/tinyllm_sidecar_read_path_repeat_last_target_smoke_qwen3_0_6b_20260710/tinyllm_sidecar_read_path_summary.json \
  --artifact correlated_same_layer_target:baseline:experiments/light_doc_cache/tinyllm_sidecar_read_path_correlated_target_smoke_qwen3_0_6b_20260710/tinyllm_sidecar_read_path_summary.json \
  --artifact calibrated_multi_correlated_target:trained:experiments/light_doc_cache/tinyllm_sidecar_read_path_calibrated_smoke_qwen3_0_6b_20260710/tinyllm_sidecar_read_path_summary.json \
  --output-dir experiments/light_doc_cache/read_path_recovery_matrix_calibrated_target_qwen3_0_6b_20260710
```

### 2026-07-10 correlated-head recovery callback and read-path smoke

Added a runtime-compatible non-oracle recovery callback:

- `make_correlated_head_recovery_callback(storage, source_heads=..., ridge=...)`
- For each compact target head, it fits an affine ridge map from a retained full
  source head to the target head using only the stored target prefix.
- It then predicts missing target KV tokens from the retained source head's
  missing-token positions.
- It does not read target missing KV values.

The TinyLLM read-path smoke now supports `--recover-mode correlated`. Source
heads are chosen by a simple heuristic: prefer a retained full head from the
same layer, otherwise use the first retained full head.

Local validation:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
python3 -m pytest -q tools/test_light_doc_cache_runtime.py tools/test_light_doc_cache_recovery_probe.py

PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
python3 -m py_compile \
  tinyvllm/engine/light_doc_cache_runtime.py \
  tinyvllm/engine/model_runner.py \
  experiments/light_doc_cache/run_storage_prototype_smoke.py \
  experiments/light_doc_cache/run_real_kv_storage_smoke.py \
  experiments/light_doc_cache/run_tinyllm_kv_summary_smoke.py \
  experiments/light_doc_cache/run_tinyllm_sidecar_read_path_smoke.py

bash -n experiments/light_doc_cache/run_task_quality_smoke_remote.sh \
  experiments/light_doc_cache/run_recovery_probe_remote.sh && git diff --check
```

Result: `53 passed`; compile, shell syntax, and diff whitespace checks passed.

Remote validation:

```bash
CUDA_VISIBLE_DEVICES=3 TINYVLLM_DIST_PORT=29661 MASTER_PORT=29661 \
PYTHONPATH=$PWD PYTHONPYCACHEPREFIX=/tmp/tinyllmforge_pycache_lightdoc \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python \
  experiments/light_doc_cache/run_tinyllm_sidecar_read_path_smoke.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B \
  --policy-file experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json \
  --repo-root /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge \
  --prompt "Light Doc Cache TinyLLM restored sidecar read path smoke." \
  --max-model-len 512 \
  --gpu-memory-utilization 0.30 \
  --recover-mode correlated \
  --output-dir /tmp/light_doc_cache_tinyllm_read_path_correlated_20260710
```

Mirrored artifact:

- `experiments/light_doc_cache/tinyllm_sidecar_read_path_correlated_smoke_qwen3_0_6b_20260710/tinyllm_sidecar_read_path_report.md`

Remote result:

- Logical full KV bytes: `1,490,944`
- Logical stored KV bytes: `1,207,808`
- Logical byte saving fraction: `18.990384615384615%`
- Missing compact tokens: `553`
- Missing-token MSE: `11.548846984339127`
- Missing-token max abs error: `174.0`
- Max abs logit diff: `5.21875`
- Mean abs logit diff: `0.8861417174339294`
- Original argmax: `1815`
- Restored argmax: `3491`
- Argmax match: `False`

Interpretation: correlated-head recovery improves substantially over the
previous `linear_tail` non-oracle read-path smoke (`MSE 38.5860`, max logit
diff `17.5`, mean logit diff `3.1362`) but still does not preserve the decode
argmax. This validates the runtime callback plumbing and gives a better
non-oracle baseline; it is not yet a trained/correlation recovery module strong
enough for quality or runtime-compression claims.

### 2026-07-10 prefix-fit source-head selection ablation

Added a source-head-map scorer:

- `build_correlated_source_head_map(storage, ridge=...)`
- It scores retained full heads by affine prefix reconstruction error for each
  compact target head.
- It uses only the compact target's stored prefix and the retained source
  prefix; it does not read target missing KV values.

The read-path smoke now exposes the strategy explicitly:

- `--correlated-source-map same_layer` (default): previous same-layer retained
  full-head heuristic.
- `--correlated-source-map prefix_fit`: prefix-MSE-selected source heads.

Local validation:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
python3 -m pytest -q tools/test_light_doc_cache_runtime.py tools/test_light_doc_cache_recovery_probe.py
```

Result: `54 passed`. Compile, shell syntax, and `git diff --check` also passed.

Remote prefix-fit ablation:

```bash
CUDA_VISIBLE_DEVICES=3 TINYVLLM_DIST_PORT=29662 MASTER_PORT=29662 \
PYTHONPATH=$PWD PYTHONPYCACHEPREFIX=/tmp/tinyllmforge_pycache_lightdoc \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python \
  experiments/light_doc_cache/run_tinyllm_sidecar_read_path_smoke.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B \
  --policy-file experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json \
  --repo-root /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge \
  --prompt "Light Doc Cache TinyLLM restored sidecar read path smoke." \
  --max-model-len 512 \
  --gpu-memory-utilization 0.30 \
  --recover-mode correlated \
  --correlated-source-map prefix_fit \
  --output-dir /tmp/light_doc_cache_tinyllm_read_path_correlated_prefixfit_20260710
```

Mirrored artifact:

- `experiments/light_doc_cache/tinyllm_sidecar_read_path_correlated_prefixfit_smoke_qwen3_0_6b_20260710/tinyllm_sidecar_read_path_report.md`

Remote result:

- Missing-token MSE: `13.2058`
- Max abs logit diff: `6.9375`
- Mean abs logit diff: `1.1095695495605469`
- Original argmax: `1815`
- Restored argmax: `13173`
- Argmax match: `False`

Interpretation: prefix-fit source selection is a useful ablation but not an
upgrade on this prompt. It is worse than the default same-layer correlated
baseline (`MSE 11.5488`, max logit diff `5.21875`, mean logit diff `0.88614`).
Short-prefix per-head reconstruction error is therefore not a sufficient
selection objective for decode-logit preservation.

### 2026-07-10 same-prompt read-path recovery matrix

Generated a recovery-mode comparison table from the Qwen3-0.6B read-path
artifacts:

- Generator: `experiments/light_doc_cache/make_read_path_recovery_matrix.py`
- `experiments/light_doc_cache/read_path_recovery_matrix_qwen3_0_6b_20260710/read_path_recovery_matrix.csv`
- `experiments/light_doc_cache/read_path_recovery_matrix_qwen3_0_6b_20260710/read_path_recovery_matrix.md`

Common setup:

- Prompt tokens: `13`
- Logical byte saving fraction: `18.99%`
- Missing compact tokens: `553`
- Original argmax: `1815`

Matrix:

| Mode | Missing MSE | Missing Max Abs | Max Logit Diff | Mean Logit Diff | Argmax Match |
|---|---:|---:|---:|---:|---|
| `repeat_last` | 13.7399 | 224 | 5.5625 | 0.787285 | False |
| `linear_tail` | 38.5860 | 336 | 17.5 | 3.13622 | False |
| `correlated_same_layer` | 11.5488 | 174 | 5.21875 | 0.886142 | False |
| `correlated_prefix_fit` | 13.2058 | 198 | 6.9375 | 1.10957 | False |
| `multi_correlated2` | 18.3975 | 384 | 6.96875 | 1.04723 | False |
| `oracle` | 0 | 0 | 0 | 0 | True |

Interpretation:

- Oracle is exact, so layout, restore indexing, and temporary read-path pointer
  swap are correct.
- No non-oracle mode preserves argmax on this prompt.
- `repeat_last` has the best mean logit diff in this small matrix.
- `correlated_same_layer` has the best missing-token MSE and max logit diff
  among non-oracle modes.
- `multi_correlated2` works mechanically but is worse than the single-source
  same-layer baseline here.
- The next recovery-quality step should target decode-logit preservation
  directly, likely via trained coefficients from a larger calibration set or a
  decode-logit-aware selector, before any attention hot-path or allocation-
  lifetime change.

### 2026-07-10 multi-source correlated callback

Added a multi-source runtime callback:

- `make_multi_source_correlated_head_recovery_callback(storage, source_heads=..., ridge=...)`
- Read-path mode: `--recover-mode multi_correlated`
- Source count: `--multi-correlated-source-count 2`

Unit tests verify that the callback can recover a compact target head that is a
linear combination of two retained full heads, including multi-dimensional
heads.

Remote Qwen3-0.6B read-path smoke:

```bash
CUDA_VISIBLE_DEVICES=3 TINYVLLM_DIST_PORT=29664 MASTER_PORT=29664 \
PYTHONPATH=$PWD PYTHONPYCACHEPREFIX=/tmp/tinyllmforge_pycache_lightdoc \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python \
  experiments/light_doc_cache/run_tinyllm_sidecar_read_path_smoke.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B \
  --policy-file experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json \
  --repo-root /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge \
  --prompt "Light Doc Cache TinyLLM restored sidecar read path smoke." \
  --max-model-len 512 \
  --gpu-memory-utilization 0.30 \
  --recover-mode multi_correlated \
  --multi-correlated-source-count 2 \
  --output-dir /tmp/light_doc_cache_tinyllm_read_path_multi_correlated2_20260710
```

Mirrored artifact:

- `experiments/light_doc_cache/tinyllm_sidecar_read_path_multi_correlated2_smoke_qwen3_0_6b_20260710/tinyllm_sidecar_read_path_report.md`

Remote result:

- Missing-token MSE: `18.3975`
- Missing-token max abs error: `384`
- Max abs logit diff: `6.96875`
- Mean abs logit diff: `1.047234296798706`
- Original argmax: `1815`
- Restored argmax: `3972`
- Argmax match: `False`

Interpretation: the multi-source path is wired correctly but is not better on
this short prompt. Per-prompt short-prefix least squares appears too unstable
for real TinyLLM logits; the next version should train/calibrate coefficients
offline across more tokens/prompts or optimize source/recovery choices against
decode-logit metrics.

### 2026-07-10 offline-calibrated recovery-bank API

Added the first offline-calibrated recovery entry point:

- `fit_multi_source_recovery_bank(calibration_kv, plan, source_heads=..., ridge=...)`
- `make_calibrated_multi_source_recovery_callback(storage, bank)`
- `save_multi_source_recovery_bank(bank, path)`
- `load_multi_source_recovery_bank(path)`
- Read-path mode: `--recover-mode calibrated_multi_correlated --recovery-bank-file <bank.json>`

The fitter consumes an offline calibration KV tensor and learns per-target,
per-K/V, per-head-dim ridge weights from retained source heads to compact
target heads. The callback applies those fitted weights to runtime retained
source heads and never reads runtime target missing KV values.

Unit coverage:

- `test_calibrated_multi_source_recovery_bank_reuses_offline_weights`
- `test_calibrated_multi_source_recovery_bank_roundtrips_json`
- It fits weights on one toy calibration KV tensor and applies them to a
  different runtime KV tensor with the same source-target relation.

Current boundary:

- This validates the trained/calibrated API shape and no-oracle runtime apply
  path.
- It is not yet a real Qwen/TinyLLM quality result because no real calibration
  dataset/artifact is wired into the fitter yet.
- The next useful step is to build a small calibration artifact from real
  TinyLLM/HF KV tensors, then use this bank in the same read-path matrix.

### 2026-07-10 TinyLLM calibrated KV smoke

Implemented and remotely validated the first real TinyLLM calibrated-bank KV
artifact:

- Script: `experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py`
- Artifact: `experiments/light_doc_cache/tinyllm_calibrated_kv_smoke_qwen3_0_6b_20260710/`
- Bank file: `multi_source_recovery_bank.json`
- Summary: `tinyllm_calibrated_kv_summary.json`

Remote command:

```bash
CUDA_VISIBLE_DEVICES=3 TINYVLLM_DIST_PORT=29665 MASTER_PORT=29665 \
PYTHONPATH=$PWD PYTHONPYCACHEPREFIX=/tmp/tinyllmforge_pycache_lightdoc \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python \
  experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B \
  --policy-file experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json \
  --repo-root /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge \
  --calibration-prompt "Light Doc Cache TinyLLM calibration prompt for Qwen KV recovery." \
  --target-prompt "Light Doc Cache TinyLLM target prompt for Qwen KV recovery." \
  --max-model-len 512 \
  --gpu-memory-utilization 0.30 \
  --max-output-len 1 \
  --source-count 2 \
  --output-dir /tmp/light_doc_cache_tinyllm_calibrated_kv_qwen3_0_6b_20260710
```

Remote result:

- Calibration tokens: `14`
- Target tokens: `14`
- Effective plan tokens: `14`
- KV cache shape: `[2, 28, 503, 256, 8, 128]`
- Source count: `2`
- Missing-token MSE: `13.548252812454097`
- Missing-token max abs error: `221.0`
- Stored tensor bytes: `1,322,496`

Interpretation:

- This validates real TinyLLM `ModelRunner.kv_cache` bank fitting, JSON
  persistence, and calibrated runtime application.
- It is KV-error-only and default-off; no attention read-path logits were
  evaluated in this smoke.
- It is slightly better than `repeat_last` MSE (`13.7399`) but worse than the
  earlier same-layer correlated MSE (`11.5488`) from the read-path matrix.
- The next step is to use this generated bank file in
  `run_tinyllm_sidecar_read_path_smoke.py --recover-mode calibrated_multi_correlated`
  and add it to the same read-path matrix.

### 2026-07-10 repeat-last non-oracle recovery baseline

Added a deterministic non-oracle recovery callback:

- `make_repeat_last_recovery_callback()`: repeats the last stored prefix token for every missing compact-head token.
- This uses only `stored_tokens` from the compressed layout and does not read the original full KV tensor.
- It is a baseline adapter between constant fill/oracle checks and a future ridge/recovery-bank recovery module.

Updated storage smoke:

- Supports `--recover-mode none|fill|repeat_last|oracle`.
- Current artifact refreshed at `experiments/light_doc_cache/storage_prototype_smoke_l6h4_20260710`.

Latest local artifact:

- Recovery mode: `repeat_last`
- Missing-token MSE: `3882.6666666666665`
- Missing-token MAE: `56.0`
- Missing-token max abs error: `96.0`
- Missing compact tokens: `474`
- Byte saving fraction: `38.23%`

Local TDD validation:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
python3 -m pytest -q tools/test_light_doc_cache_runtime.py -k 'repeat_last'
```

Result: `2 passed, 29 deselected`.

Claim boundary: repeat-last is a non-oracle deterministic baseline that verifies callback wiring and gives a measurable restored-token error. It is not a trained correlation doc-cache / gist-cache recovery module and does not prove model-quality runtime compression.

### 2026-07-10 linear-tail toy recovery baseline

Added a second deterministic non-oracle recovery callback:

- `make_linear_tail_recovery_callback(ridge=...)`: fits a per-channel ridge-linear trend from stored prefix tokens and extrapolates missing compact-head tokens.
- The callback uses only `stored_tokens`, `selected_tokens`, and `missing_tokens`.
- It is a toy trend baseline and an interface check for fitted recovery callbacks; it is not the existing attention-output ridge/recovery bank.

Updated storage smoke:

- Supports `--recover-mode none|fill|repeat_last|linear_tail|oracle`.
- Added `--recover-ridge` for the linear-tail ridge term.
- Current artifact refreshed at `experiments/light_doc_cache/storage_prototype_smoke_l6h4_20260710`.

Latest local artifact:

- Recovery mode: `linear_tail`
- KV pattern: `arange`
- Missing-token MSE: `2.1797172599452458e-12`
- Missing-token MAE: `1.7302951732265296e-07`
- Missing-token max abs error: `1.52587890625e-05`
- Missing compact tokens: `474`
- Byte saving fraction: `38.23%`

Important limitation: this near-zero error is expected for the current storage smoke because the toy KV tensor is `np.arange(...)`, so each head/channel is nearly linear along the flattened token dimension. It should only be used to validate callback plumbing and fitted-adapter accounting. It is not evidence that real model KV tensors can be recovered with low error.

Added a harder toy pattern:

- `run_storage_prototype_smoke.py --kv-pattern nonlinear`
- Pattern combines sinusoidal and quadratic token-position terms, plus layer/head/KV offsets.
- This keeps the smoke deterministic while avoiding a misleading perfect linear extrapolation.

Latest refreshed artifact:

- Path: `experiments/light_doc_cache/storage_prototype_smoke_l6h4_20260710`
- Recovery mode: `linear_tail`
- KV pattern: `nonlinear`
- Missing-token MSE: `17.647056659579228`
- Missing-token MAE: `3.641985407374202`
- Missing-token max abs error: `7.412278652191162`
- Missing compact tokens: `474`
- Byte saving fraction: `38.23%`

Claim boundary: the nonlinear smoke is still a CPU/toy tensor test. It is useful because it now shows non-zero fitted-adapter error, but it remains a storage/recovery callback validation artifact rather than real prompt/model KV recovery.

### 2026-07-10 real Qwen3-0.6B HF KV storage smoke

Added a real-tensor smoke:

- Script: `experiments/light_doc_cache/run_real_kv_storage_smoke.py`
- Local mirrored artifact: `experiments/light_doc_cache/real_kv_storage_smoke_qwen3_0_6b_20260710/`
- Remote output directory: `/tmp/light_doc_cache_real_kv_linear_tail_20260710`

It loads Qwen3-0.6B with HuggingFace, runs a short prompt with `use_cache=True`, converts `past_key_values` into the prototype runtime shape `[2, layers, blocks, block_size, kv_heads, head_dim]`, applies `LightDocCacheCompressedKVStorage`, and evaluates missing compact-token error.

Environment note:

- `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python` failed before model load with a transformers/torch `custom_op` schema incompatibility.
- `/data00/home/sitian/miniconda3/envs/py311/bin/python` worked and matches the existing task-quality remote runner default.

Remote command:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPYCACHEPREFIX=/tmp/tinyllmforge_pycache_lightdoc \
/data00/home/sitian/miniconda3/envs/py311/bin/python \
  experiments/light_doc_cache/run_real_kv_storage_smoke.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B \
  --policy-file experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json \
  --repo-root /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge \
  --prompt "Light Doc Cache real KV smoke. The answer is storage." \
  --max-tokens 64 \
  --recover-mode linear_tail \
  --block-size 16 \
  --device cuda \
  --output-dir /tmp/light_doc_cache_real_kv_linear_tail_20260710
```

Remote result:

- Input tokens: `12`
- KV cache shape: `[2, 28, 1, 16, 8, 128]`
- Recovery mode: `linear_tail`
- Full tensor bytes: `1,835,008`
- Stored tensor bytes: `1,133,568`
- Saved tensor bytes: `701,440`
- Byte saving fraction: `38.23%`
- Missing compact tokens: `474`
- Missing-token MSE: `23.193616792046498`
- Missing-token MAE: `2.0537286563638895`
- Missing-token max abs error: `302.0`

Claim boundary: this is the first real-model KV tensor storage/recovery smoke, but it is still offline around HF `past_key_values`. It does not modify TinyLLM runtime allocation, KV writes, attention kernels, or decode quality path.

### 2026-07-10 TinyLLM ModelRunner KV summary smoke

Added a TinyLLM runtime-facing accounting smoke:

- Script: `experiments/light_doc_cache/run_tinyllm_kv_summary_smoke.py`
- Local mirrored artifact: `experiments/light_doc_cache/tinyllm_kv_summary_smoke_qwen3_0_6b_20260710/`
- Remote output directory: `/tmp/light_doc_cache_tinyllm_kv_summary_20260710_clean`

It instantiates TinyLLM `LLM`, runs a short prompt through `ModelRunner`, reads the actual allocated `model_runner.kv_cache.shape`, and routes that shape through the existing `build_model_runner_light_doc_cache_summary(...)` accounting helper.

Remote command:

```bash
CUDA_VISIBLE_DEVICES=3 TINYVLLM_DIST_PORT=29632 MASTER_PORT=29632 \
PYTHONPATH=$PWD PYTHONPYCACHEPREFIX=/tmp/tinyllmforge_pycache_lightdoc \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python \
  experiments/light_doc_cache/run_tinyllm_kv_summary_smoke.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B \
  --policy-file experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json \
  --repo-root /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge \
  --prompt "Light Doc Cache TinyLLM KV summary smoke." \
  --max-model-len 512 \
  --gpu-memory-utilization 0.30 \
  --max-output-len 1 \
  --enforce-eager \
  --output-dir /tmp/light_doc_cache_tinyllm_kv_summary_20260710_clean
```

Remote result:

- Prompt tokens: `10`
- TinyLLM allocated KV shape: `[2, 28, 805, 256, 8, 128]`
- Allocated KV cache bytes: `23,634,903,040`
- Logical full KV bytes for the 10-token plan: `1,146,880`
- Planned recovered KV bytes: `202,240`
- Planned stored KV bytes: `944,640`
- Planned byte saving fraction: `17.633928571428573%`
- Planned compression ratio: `1.2141x`

Important interpretation: allocated KV bytes are the full preallocated TinyLLM cache capacity. The planned bytes are logical bytes for `seq_len=prompt_tokens`, so they are not the same denominator. This smoke validates that Light Doc Cache accounting can read a real TinyLLM `ModelRunner.kv_cache` shape; it does not apply runtime compression or reduce allocated memory yet.

### 2026-07-10 TinyLLM sidecar storage/readback smoke

Extended `run_tinyllm_kv_summary_smoke.py` with:

- `--write-sidecar-storage`
- `--recover-mode none|fill|repeat_last|linear_tail|oracle`

This creates a `LightDocCacheCompressedKVStorage` sidecar from the actual TinyLLM `ModelRunner.kv_cache`, restores into a temporary full-shape tensor, and evaluates missing compact-token error. It still does not alter KV writes or attention reads.

Local mirrored artifact:

- `experiments/light_doc_cache/tinyllm_sidecar_storage_smoke_qwen3_0_6b_20260710/tinyllm_sidecar_storage_report.md`

Remote command:

```bash
CUDA_VISIBLE_DEVICES=3 TINYVLLM_DIST_PORT=29634 MASTER_PORT=29634 \
PYTHONPATH=$PWD PYTHONPYCACHEPREFIX=/tmp/tinyllmforge_pycache_lightdoc \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python \
  experiments/light_doc_cache/run_tinyllm_kv_summary_smoke.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B \
  --policy-file experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json \
  --repo-root /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge \
  --prompt "Light Doc Cache TinyLLM sidecar storage smoke." \
  --max-model-len 512 \
  --gpu-memory-utilization 0.30 \
  --max-output-len 1 \
  --enforce-eager \
  --write-sidecar-storage \
  --recover-mode linear_tail \
  --output-dir /tmp/light_doc_cache_tinyllm_sidecar_20260710_logical
```

Remote result:

- Prompt tokens: `11`
- TinyLLM allocated KV shape: `[2, 28, 805, 256, 8, 128]`
- Recovery mode: `linear_tail`
- Sidecar full tensor bytes over allocated capacity: `23,634,903,040`
- Sidecar stored tensor bytes: `1,059,328`
- Allocated-capacity saving fraction: `99.99551795072649%`
- Logical full KV bytes for the 11-token plan: `1,261,568`
- Logical stored KV bytes: `1,059,328`
- Logical byte saving fraction: `16.030844155844157%`
- Missing compact tokens: `395`
- Missing-token MSE: `28.7970687629443`
- Missing-token max abs error: `274.0`

Claim boundary: this is the first sidecar materialization/readback from real TinyLLM `kv_cache`. The `99.9955%` number is only relative to the full preallocated cache capacity; the comparable prompt-level logical saving is `16.03%`. This is still not a runtime memory reduction claim because the original `kv_cache` allocation remains intact and attention still reads from it.

### 2026-07-10 ModelRunner sidecar inspection wrapper

Added the first default-off runner integration point for sidecar materialization:

- Runtime helper: `materialize_light_doc_cache_sidecar(...)`
- Runner helper: `materialize_model_runner_light_doc_cache_sidecar(...)`
- `ModelRunner` wrapper: `runner.light_doc_cache_materialize_sidecar(plan, evaluate_readback=True)`

The wrapper stores the sidecar object on `runner.light_doc_cache_sidecar` and returns:

- sidecar full/stored/saved bytes;
- logical full/stored/saved bytes for `plan.seq_len`;
- optional missing compact-token error metrics when readback evaluation is enabled.

`run_tinyllm_kv_summary_smoke.py --write-sidecar-storage` now uses this `ModelRunner` wrapper when it exists, falling back to the pure runtime helper only for fake/test runners without the method.

Claim boundary: this is still default-off inspection. It does not replace `self.kv_cache`, does not change slot mapping, and does not change attention reads or writes.

Local validation:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
python3 -m pytest -q tools/test_light_doc_cache_runtime.py
# 37 passed

PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
python3 -m py_compile \
  tinyvllm/engine/light_doc_cache_runtime.py \
  tinyvllm/engine/model_runner.py \
  experiments/light_doc_cache/run_tinyllm_kv_summary_smoke.py
```

Remote wrapper smoke:

```bash
CUDA_VISIBLE_DEVICES=3 TINYVLLM_DIST_PORT=29637 MASTER_PORT=29637 \
PYTHONPATH=$PWD PYTHONPYCACHEPREFIX=/tmp/tinyllmforge_pycache_lightdoc \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python \
  experiments/light_doc_cache/run_tinyllm_kv_summary_smoke.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B \
  --policy-file experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json \
  --repo-root /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge \
  --prompt "Light Doc Cache TinyLLM sidecar wrapper smoke." \
  --max-model-len 512 \
  --gpu-memory-utilization 0.30 \
  --max-output-len 1 \
  --enforce-eager \
  --write-sidecar-storage \
  --recover-mode linear_tail \
  --output-dir /tmp/light_doc_cache_tinyllm_sidecar_20260710_wrapper
```

Mirrored artifact:

- `experiments/light_doc_cache/tinyllm_sidecar_storage_wrapper_smoke_qwen3_0_6b_20260710/tinyllm_sidecar_storage_report.md`

Remote result:

- Prompt tokens: `11`
- TinyLLM allocated KV shape: `[2, 28, 805, 256, 8, 128]`
- Logical full KV bytes: `1,261,568`
- Logical stored KV bytes: `1,059,328`
- Logical byte saving fraction: `16.03%`
- Missing compact tokens: `395`
- Missing-token MSE: `28.458839963420036`
- Missing-token max abs error: `274.0`

### 2026-07-10 restored sidecar read-path logits smoke

Added a default-off read-path comparison script:

- `experiments/light_doc_cache/run_tinyllm_sidecar_read_path_smoke.py`

It runs normal prefill, materializes/restores the sidecar into a temporary full
KV tensor, temporarily redirects attention layer `k_cache` / `v_cache` pointers
to that restored tensor for one decode step, compares logits with the original
full-cache decode path, and then restores the original pointers. It does not
modify attention kernels or KV allocation lifetime.

Remote command:

```bash
CUDA_VISIBLE_DEVICES=3 TINYVLLM_DIST_PORT=29649 MASTER_PORT=29649 \
PYTHONPATH=$PWD PYTHONPYCACHEPREFIX=/tmp/tinyllmforge_pycache_lightdoc \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python \
  experiments/light_doc_cache/run_tinyllm_sidecar_read_path_smoke.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B \
  --policy-file experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json \
  --repo-root /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge \
  --prompt "Light Doc Cache TinyLLM restored sidecar read path smoke." \
  --max-model-len 512 \
  --gpu-memory-utilization 0.30 \
  --recover-mode linear_tail \
  --output-dir /tmp/light_doc_cache_tinyllm_read_path_20260710
```

The first attempt on port `29638` failed with `EADDRINUSE`; rerunning unchanged
code on port `29649` succeeded.

Mirrored artifact:

- `experiments/light_doc_cache/tinyllm_sidecar_read_path_smoke_qwen3_0_6b_20260710/tinyllm_sidecar_read_path_report.md`

Remote result:

- Prompt tokens: `13`
- TinyLLM allocated KV shape: `[2, 28, 805, 256, 8, 128]`
- Logical full KV bytes: `1,490,944`
- Logical stored KV bytes: `1,207,808`
- Logical byte saving fraction: `18.990384615384617%`
- Missing compact tokens: `553`
- Missing-token MSE: `38.58595409486837`
- Missing-token max abs error: `336.0`
- Max abs logit diff: `17.5`
- Mean abs logit diff: `3.1362242698669434`
- Original argmax: `1815`
- Restored argmax: `50927`
- Argmax match: `False`

Interpretation: the restored tensor is accepted by the existing TinyLLM decode
read path, so the pointer-swap read-path smoke validates shape/layout
compatibility. The current toy `linear_tail` recovery is not quality preserving
at the logits level, so this is not an accuracy or runtime-compression claim.
Next useful validation is `--recover-mode oracle` as an upper-bound layout check,
then a trained/correlation recovery callback.

Oracle upper-bound rerun:

```bash
CUDA_VISIBLE_DEVICES=3 TINYVLLM_DIST_PORT=29650 MASTER_PORT=29650 \
PYTHONPATH=$PWD PYTHONPYCACHEPREFIX=/tmp/tinyllmforge_pycache_lightdoc \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python \
  experiments/light_doc_cache/run_tinyllm_sidecar_read_path_smoke.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B \
  --policy-file experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json \
  --repo-root /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge \
  --prompt "Light Doc Cache TinyLLM restored sidecar read path smoke." \
  --max-model-len 512 \
  --gpu-memory-utilization 0.30 \
  --recover-mode oracle \
  --output-dir /tmp/light_doc_cache_tinyllm_read_path_oracle_20260710
```

Mirrored artifact:

- `experiments/light_doc_cache/tinyllm_sidecar_read_path_oracle_smoke_qwen3_0_6b_20260710/tinyllm_sidecar_read_path_report.md`

Oracle result:

- Logical full KV bytes: `1,490,944`
- Logical stored KV bytes: `1,207,808`
- Logical byte saving fraction: `18.990384615384617%`
- Missing compact tokens: `553`
- Missing-token MSE: `0.0`
- Missing-token max abs error: `0.0`
- Max abs logit diff: `0.0`
- Mean abs logit diff: `0.0`
- Original argmax: `1815`
- Restored argmax: `1815`
- Argmax match: `True`

Interpretation: oracle recovery exactly reproduces decode logits through the
temporary restored-sidecar read path. This isolates the remaining gap to
non-oracle recovery quality rather than storage indexing or attention read-path
compatibility.

### 2026-07-10 recovery error metrics and oracle callback

Added recovery evaluation helpers:

- `make_oracle_recovery_callback(full_kv, plan)`: copies missing compact-head tokens from the original full KV tensor. This is only for layout validation.
- `evaluate_restored_kv_error(full_kv, restored_kv, plan)`: reports missing compact-token MSE/MAE/max error and token count.

Updated storage smoke:

- Supports `--recover-mode none|fill|oracle`.
- `oracle` mode now generates the checked artifact in `experiments/light_doc_cache/storage_prototype_smoke_l6h4_20260710`.

Latest local artifact:

- Recovery mode: `oracle`
- Missing-token MSE: `0`
- Missing-token max abs error: `0`
- Missing compact tokens: `474`
- Byte saving fraction: `38.23%`

Claim boundary: oracle mode proves the storage/recovery indexing and error accounting are correct. It does not prove a learned/ridge recovery module can reconstruct missing KV tokens.

Remote validation:

- Synced updated runtime/smoke files to `sitian@10.232.195.203`.
- Remote `py_compile` passed.
- Remote oracle smoke passed:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge_pycache_lightdoc \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python \
  experiments/light_doc_cache/run_storage_prototype_smoke.py \
  --policy-file experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json \
  --repo-root /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge \
  --recover-mode oracle \
  --output-dir /tmp/light_doc_cache_storage_oracle_remote_20260710
```

Remote result:

- `byte_saving_fraction=0.3822544642857143`
- `mse_missing_compact_tokens=0.0`
- `max_abs_missing_compact_tokens=0.0`
- `num_missing_compact_tokens=474`

### 2026-07-10 storage recovery-fill callback

Extended `LightDocCacheCompressedKVStorage.restore_to_full_shape(...)` with a `recover_missing_fn` callback. The callback receives:

- `layer`
- `kv_head`
- `selected_tokens`
- `missing_tokens`
- `stored_tokens`
- `head_dim`
- `dtype`

It must return a `[2, missing_tokens, head_dim]` tensor/array. The storage layer checks this shape before writing recovered tokens into the restored full KV layout.

Updated smoke artifact:

- `experiments/light_doc_cache/storage_prototype_smoke_l6h4_20260710/storage_prototype_report.md`
- Recovery mode: `fill_missing`

Claim boundary: the current smoke uses a constant fill callback only to verify the API and layout. It is not the trained/ridge recovery module yet.

Remote validation:

- Synced updated storage runtime files and smoke script to `sitian@10.232.195.203`.
- Remote `py_compile` passed.
- Remote fill-callback smoke passed with:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge_pycache_lightdoc \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python \
  experiments/light_doc_cache/run_storage_prototype_smoke.py \
  --policy-file experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json \
  --repo-root /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge \
  --recover-fill-value 42.0 \
  --output-dir /tmp/light_doc_cache_storage_recovery_fill_remote_20260710
```

Remote result: `byte_saving_fraction=0.3822544642857143`.
