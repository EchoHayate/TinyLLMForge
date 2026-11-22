# 新校准 Gate G3：72 次 Qwen3-8B 配对校准诊断最终完成与独立核验结果

日期：2026-11-22，Asia/Shanghai。

状态：**已完成全部 72 次 Qwen3-8B 贪婪生成（Phase 1）与全部 18 批 576 项 `bwrap`+`seccomp` 隔离评分（Phase 2），并通过独立离线全量重建复核（`APPROVE`，0 P0 / 0 P1 / 0 P2）；校准状态正式达到 `DEVELOPMENT_DIAGNOSTIC_USABLE`（零协议矛盾、零缺失项、零校准不达标项）。**

## 1. 核心诊断结论与各条件表现（12 题 × 8 用例 = 每候选满分 8/8）

| 指标 | `base`（无示例提示） | `text`（拼接可信监督示例提示） |
| --- | --- | --- |
| 首轮 `initial` 全通过（`PASS`） | **`6 / 12`**（`C01, C05, C07, C09, C10, C11`） | **`4 / 12`**（`C01, C05, C07, C11`） |
| 首轮 `initial` 语义错误（`SEMANTIC_FAIL`） | **`6 / 12`**（`C02, C03, C04, C06, C08, C12`） | **`8 / 12`**（`C02, C03, C04, C06, C08, C09, C10, C12`） |
| 首轮执行失败 / 未知 / 无效证据 | `0 / 0 / 0` | `0 / 0 / 0` |
| 首轮语义错误被公开用例揭示数 | **`6 / 6`（覆盖全部 4 类机制）** | **`8 / 8`（覆盖全部 4 类机制）** |
| 二轮 `feedback`（含公开测试状态）全通过 | **`6 / 12`**（纠错 `0`，退化 `0`） | **`6 / 12`**（纠错 `2`：`C09, C10`；退化 `0`） |
| 二轮 `retry`（隐去测试状态对照）全通过 | **`6 / 12`**（纠错 `0`，退化 `0`） | **`6 / 12`**（纠错 `2`：`C09, C10`；退化 `0`） |
| **`feedback` 相对 `retry` 配对净增（`net_gain`）** | **`0`**（`0` 赢 / `0` 输） | **`0`**（`0` 赢 / `0` 输） |

### 关键实验发现

1. **新题库难度校准完全命中目标区间（`DEVELOPMENT_DIAGNOSTIC_USABLE`）**：
   - 彻底解决了 v1 开发集 `16/16` 满分天花板饱和问题：`base` 与 `text` 均既有稳定通过题（`6` 题与 `4` 题），又有充足的真实语义错误空间（`6` 题与 `8` 题，为后续权重适应保留了 `6–8` 题的净提升空间），且全部 `72/72` 份生成代码 100% 满足单函数 AST 语法与签名约束。
2. **长文本上下文示例（`text`）暴露出显著的推理耗时膨胀与首轮干扰**：
   - 在完全相同的 36 次贪婪生成任务上，`text` 输入 token 数达 `91,230`（`base` 为 `20,687`，`4.41×`），`eager` 无缓存精确前向 GPU 耗时达 **`2,031.40` 秒**（`base` 仅 **`378.23` 秒**，慢 **`5.37×`**），峰值保留显存达 `78.45 GiB`（比 `base` 高 `11.25 GiB`）；
   - 更关键的是，拼接旧版简单监督示例后，`text` 首轮反而在 `base` 原本一次做对的 `C09`、`C10` 两题上受长上下文干扰出错（首轮从 `6/12` 跌至 `4/12`），二轮才恢复至 `6/12`（最终仍未超过 `base`）。
3. **配对 `retry` 对照排除了“假阳性二轮反馈收益”**：
   - 在 `text` 组中，`feedback` 从首轮 `4/12` 升至 `6/12`（修好 `C09`、`C10`）；若无 `retry` 对照，极易误判为“公开测试反馈带来 +2 题收益”。但同首轮分叉且不提供测试结果的 `retry` 同样修好了 `C09` 与 `C10`（`4/12 -> 6/12`），证明这两题完全来自多一轮自检重答，`visible_tests` 粗粒度状态反馈的真实净增为 **`0`**。

## 2. 分层实测成本汇总

| 阶段 / 成本项 | 实测数值 |
| --- | ---: |
| Qwen3-8B 12 文件 SHA256 核验 | `27.624` 秒 |
| 模型与分词器加载（BF16，1× A100 80GB） | `22.099` 秒（加载峰值 `15.26 GiB` allocated） |
| `base` 36 次生成 GPU 墙钟 / 公开核验耗时 | `378.229` 秒 / `2.755` 秒 |
| `text` 36 次生成 GPU 墙钟 / 公开核验耗时 | `2,031.403` 秒 / `2.902` 秒 |
| 生成后模型权重 / 分词器 / RNG 完整性核验 | `31.497` 秒（`rng_unchanged=true`） |
| Phase 2 全部 18 批 576 项隔离执行累计耗时 / 外层墙钟 | `38.252` 秒 / `70.532` 秒 |

## 3. 封存证据与独立复核摘要

- Phase 1 封存记录：`/private/tmp/lora-recal-72-diag-runs-r3-20261121-4tyx4l88/b0dc5a77fa55f0b6a2a169f68a529673/phase1-wire.bin`（SHA256 `353172ce9a39649b2a25588dd28db48744283edd152dc8923cd35d2b1010a3ba`）
- Phase 2 完整报告：`/private/tmp/lora-recal-72-diag-p2-runs-20261122-ss6hita9/dbf831b5f7a6e0f82fd36831d34a87d1/diagnostic-report-full.json`（SHA256 `325773a55549c7a5645381468690f29fb61440d4bf3087e7344e846b63e99ca9`）
- 独立离线重建复核：`/private/tmp/lora-recal-72-diag-p2-capture-20261122-b_2sazgu/independent-72-diagnostic-postrun-review.md`（SHA256 `ea29d1e43ff15f399c7620121120471b0e149b3dd9c113124b18a1f82e4caaa4`，`APPROVE`，0 P0/P1/P2）
- 主控阶段接受回执：`/private/tmp/lora-recal-72-diag-p2-capture-20261122-b_2sazgu/72-diagnostic-acceptance.json`（SHA256 `1aeb759277affe4dd0d15f6cb7d0fc9d86d964a78f6b0522553b7b0eb9cb09c7`）

## 4. 治理边界与下一步（Gate G4：同信息 LoRA 四臂确认性对照）

按 `docs/superpowers/specs/2026-11-17-lora-development-recalibration-design.md` 第 7–8 节规定，`C01..C12` 已履行开发校准使命，不得回用为 LoRA 训练或收益验收集，v1 旧迁移/保留题亦保持封存不复用。`benchmark_eligible`、`formal_evaluation_ready`、`net_benefit_established` 保持 `false`。下一步进入新设计的 **v2 同信息 LoRA 四臂确认性对照实验**（`docs/superpowers/specs/2026-11-22-lora-v2-confirmatory-comparison-design.md`）。
