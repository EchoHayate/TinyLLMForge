# SDD 工作流详细指南

## 为什么需要 SDD？Vibe Coding 的问题

Vibe Coding（直接让 AI 写代码，不做结构化规范）会导致以下常见问题：

| 问题 | 表现 | SDD 如何解决 |
|------|------|-------------|
| 架构腐化 | 50 人能跑但 1 万人崩溃；命名/结构混乱 | Spec 预先定义架构；Constitution 强制规范 |
| 知识蒸发 | 只有 AI 知道代码怎么跑；新人无法理解 | `spec.md` + `plan.md` 形成活的文档 |
| 速度悖论 | 起步很快，但后续全是填坑；总耗时反而增加 | Spec 让你保持匀速前进，减少返工 |
| 上下文丢失 | AI 在长对话中忘记上下文；修 A 搞坏 B | 每个命令从 spec 文件读取，不依赖对话历史 |
| 沟通偏差 | 模糊需求被 AI 脑补成错误实现 | 结构化 spec 配合验收标准消除歧义 |
| 质量问题 | 没测试、没文档、代码像黑箱 | SDD 先产出设计制品，再推进实现 |

**SDD 核心理念**：在让 AI 写代码之前，用结构化的方式描述清楚要做什么。Spec 成为真相来源，代码只是它的“编译产物”。

> 更多 SDD 背景阅读：[Spec-Driven Development - AI Coding 时代的研发效能](https://bytedance.larkoffice.com/wiki/LBZSwHJOPiXiOpkrMBCcwXS1nmg)
> 真实案例分享：[TTLS 内 SDD Cases 总结](https://bytedance.sg.larkoffice.com/docx/O53odynPOorfLuxdJgNlDtFggHd)

## 当前工作流概览

在当前 TTADK 中，最常见的是三条路径：

1. **Readiness 预检查**：先用 `/adk-readiness` 评估仓库是否适合 AI 开发。
2. **标准 SDD 开发流程**：从需求到实现的主路径。
3. **快速 SDD 流程**：一步生成核心制品后快速进入实现。

### 标准 SDD 模式

```text
readiness → [sdd:brainstorm] → sdd:constitution → sdd:specify → [sdd:clarify] → sdd:plan → [sdd:erd] → sdd:tasks → [sdd:analyze] → sdd:implement → [sdd:codereview] → commit → sdd:archive
```

核心链路：

```text
Specify → Plan → Tasks → Implement
```

### 快速 SDD 模式

```text
readiness → [sdd:brainstorm] → sdd:ff → [sdd:clarify] → sdd:implement → [sdd:codereview] → commit → sdd:archive
```

核心链路：

```text
FF → Implement
```

## SDD 各阶段详解

### 阶段 0：仓库 readiness（`/adk-readiness`）

**目的**：在正式开始 SDD 前，先确认当前代码仓库是否足够 AI Friendly。

**输出**：成熟度等级、维度评分、改进建议。

**适用时机**：
- 新接手仓库。
- 想在大规模引入 AI 开发前补齐上下文、文档、测试、规范。
- 想量化仓库 readiness 改造效果。

### 阶段 1：头脑风暴（`/adk-sdd-brainstorm`，可选）

**目的**：把模糊想法、ERD、飞书资料、学习材料整理成结构化思路文档。

**输入**：文本、链接、本地文件等。

**产出**：可供团队讨论的方案文档或 Lark 文档。

**适合场景**：
- 需求仍很模糊。
- 需要先比较方案。
- 需求较大，想先拆模块再进入正式 spec。

### 阶段 2：项目原则（`/adk-sdd-constitution`）

**目的**：定义项目级原则，后续 SDD 命令都会读取并遵守。

**产出**：`docs/CONSTITUTION.md`

**典型内容**：编码规范、质量门禁、架构原则、命名约定、测试要求。

### 阶段 3：需求规格（`/adk-sdd-specify`）

**目的**：把功能想法转化成结构化规格文档。

**输入**：功能描述、飞书文档链接或更早阶段的输出。

**产出**：`specs/YYYYMMDD-feature-name/spec.md`

**要点**：
- 标准模式的核心起点。
- 会根据输入成熟度评估是否应继续或先补充信息。
- 会尽量保证 `spec.md` 覆盖原始输入中的关键信息。

### 阶段 4：需求澄清（`/adk-sdd-clarify`）

**目的**：通过交互式问答补齐规格中的模糊点，并同步下游制品。

**输入**：`spec.md` + 可选聚焦领域。

**行为**：
1. 每次提出最多 5 个关键问题。
2. 接收用户回答后更新 `spec.md`。
3. 如果 `plan.md`、`technical-design.md`、`tasks.md` 已存在，也一并同步。

**要点**：
- 可以执行多次。
- 建议在 `specify` 或 `ff` 后尽早执行。
- 如果根因是规格不清晰，优先 clarify，再继续 plan / implement。

### 阶段 5：实现计划（`/adk-sdd-plan`）

**目的**：把需求规格转成技术计划和设计骨架。

**输入**：`spec.md`

**产出**：
- `plan.md`
- `research.md`
- `data-model.md`
- `contracts/`
- `quickstart.md`

**要点**：
- 会结合代码库上下文做调研与设计。
- 强调技术决策、数据模型、接口契约和实施顺序。

### 阶段 6：技术设计（`/adk-sdd-erd`，可选）

**目的**：生成更适合评审和沟通的技术设计文档。

**输入**：通常至少需要 `spec.md`，复杂场景建议结合 `plan.md`。

**产出**：`technical-design.md`

**适用场景**：
- 改动跨模块。
- 需要结构图、时序图、ER 图。
- 需要团队评审或同步。

### 阶段 7：任务拆解（`/adk-sdd-tasks`）

**目的**：把实现计划拆成依赖有序、可执行的原子任务。

**输入**：`spec.md` + `plan.md`

**产出**：`tasks.md`

**任务特征**：
- 包含任务 ID。
- 标明依赖关系。
- 引用待改文件路径。
- 支持并行标记 `[P]`。

### 阶段 8：制品分析（`/adk-sdd-analyze`，可选）

**目的**：在实现前，对 `spec.md`、`plan.md`、`tasks.md` 做只读质量检查。

**重点检查**：
- 要求是否有遗漏。
- 任务是否覆盖所有需求。
- 术语是否漂移。
- 是否与宪章冲突。

**注意**：这是制品分析，不是代码实现或代码审查命令。

### 阶段 9：代码实现（`/adk-sdd-implement`）

**目的**：按 `tasks.md` 逐项落地代码。

**输入**：`tasks.md` + 所有规划制品。

**要点**：
- 会按依赖顺序推进任务。
- 支持指定 Phase / 任务 ID / 修正反馈。
- 适合先由流程产出任务，再让 AI 执行。

### 阶段 10：代码审查（`/adk-sdd-codereview`，可选但推荐）

**目的**：在提交前，对实现代码变更做结构化代码审查。

**输入**：当前代码 diff，或显式提供的 `base:<ref>` / Bits-Code MR / 分支目标。

**重点**：
- 与 `sdd-analyze` 不同，这里检查的是代码，不是制品。
- 默认关注正确性、测试、可维护性、规范对齐。
- 会按 diff 触发安全、性能、可靠性、契约等补充视角。

### 阶段 11：代码提交（`/adk-commit`）

**目的**：提交当前改动并附带 TTADK 追踪签名。

**要点**：
- 自动生成规范化提交信息。
- 推荐使用，以确保 AI 贡献率统计准确。

### 阶段 12：归档（`/adk-sdd-archive`）

**目的**：归档旧 feature 制品，保持 `specs/` 目录整洁。

**要点**：
- 适合功能完成后做收尾。
- 支持 dry-run / json 等模式。

## 快速模式（`/adk-sdd-ff`）

`/adk-sdd-ff` 将 `specify + plan + tasks` 融合为一步，适合：

- 需求清晰、边界稳定的功能。
- 已经过 `sdd-brainstorm` 并达成共识。
- 想快速从需求走到实现准备态。

典型流程：

```text
sdd-ff → [sdd-clarify] → sdd-implement → [sdd-codereview]
```

## 异步 Handoff 协作

如果本地不方便长时间执行，或希望异步推进，可以使用 `ttadk handoff`：

- 提交异步任务：`ttadk handoff submit "基于最新的 Spec 继续实现并产出代码"`
- 查看任务状态：`ttadk handoff list`
- 同步完成结果到本地：`ttadk handoff sync -t <task_id>`

## Readiness 检查

`/adk-readiness` 用于在进入 SDD 之前评估仓库是否足够 AI Friendly。

对应 skill：`adk-readiness`

推荐在以下时机执行：
- 新接手仓库。
- 准备系统性引入 AI Coding。
- 想量化仓库在上下文、测试、规范、CI 等方面的成熟度。

## 工作流状态流转

```text
not-started ──sdd:specify / sdd:ff──► specified-or-planned
specified-or-planned ──sdd:clarify──► specified-or-planned (refined)
specified-or-planned ──sdd:plan──► planned
planned ──sdd:erd──► designed
planned ──sdd:tasks──► tasked
designed ──sdd:tasks──► tasked
tasked ──sdd:analyze──► tasked (validated)
tasked ──sdd:implement──► implementing
implementing ──sdd:codereview──► implementing (reviewed)
implementing ──commit──► committed
committed ──sdd:archive──► archived
```

任何阶段都可以使用 `/adk-sdd-clarify` 来完善规格并同步已有下游制品。

## Feature 目录结构

一个完整的功能目录：

```text
specs/20250601-my-feature/
├── spec.md                    # 功能规格
├── plan.md                    # 实现计划
├── research.md                # 技术调研
├── data-model.md              # 数据模型定义
├── contracts/                 # API 契约
│   └── api.yaml
├── quickstart.md              # 快速上手指南
├── technical-design.md        # 技术设计文档
├── tasks.md                   # 实现任务
└── checklists/                # 可能存在的检查清单或验证产物
```

## 最佳实践

### 上下文管理
- **命令之间使用 `/clear`**：TTADK 命令是可重入的，从 spec 文件读取而非对话历史。清除上下文可以防止 token 溢出和上下文污染。
- **不要依赖对话记忆**：所有状态都持久化在 `specs/` 下的 spec 文件中。

### 质量提升
- 先执行 `/adk-readiness`：在改仓库之前先看 readiness 短板。
- 先执行 `/init` 生成 `CLAUDE.md`/`AGENTS.md`（如果不存在）：审查并自定义。
- 复杂功能先执行 `/adk-sdd-brainstorm` 或准备 ERD：先拆模块和风险点。
- 在 `specify` 或 `ff` 之后使用 `/adk-sdd-clarify`：在传播到下游制品之前捕获模糊点。
- 在 `implement` 之前使用 `/adk-sdd-analyze`：写代码前验证核心制品质量。
- 如果生成内容质量差，检查：（1）ERD/spec 是否足够详细；（2）`CLAUDE.md`/`AGENTS.md` 是否有充分的项目上下文；（3）代码库结构是否良好。

### 版本管理
- `.ttadk/` 目录：纳入版本管理（团队共享配置）。
- `docs/CONSTITUTION.md`：纳入版本管理，协作维护。
- `CLAUDE.md` / `AGENTS.md`：纳入版本管理。
- `specs/` 目录：建议纳入版本管理（定期用 `/adk-sdd-archive` 归档）。
- `specs/doc_export/`：加入 `.gitignore`（中间导出产物）。

### Monorepo
- 在仓库根目录初始化 TTADK。
- `CLAUDE.md` / `AGENTS.md` 可嵌套到子目录（用 `/init` 自动生成）。
- Constitution 保持在根目录。
- 在 ERD 中注明本次涉及的子目录。
- 多仓库开发参考：[基于 SDD 的代码仓库管理方案](https://bytedance.sg.larkoffice.com/docx/RjbrdkxmroH6uUxqSCEl5epxg1d)
