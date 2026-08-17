# TTADK 命令参考

## 基础公共命令

### /adk-help

获取 TTADK 工作流、命令和当前项目状态的帮助，也可以用来问“下一步该做什么”。

**用法**：`/adk-help [你的问题]`

**示例**：
- `/adk-help 下一步该做什么`
- `/adk-help /adk-sdd-plan 命令怎么用`
- `/adk-help readiness 报告怎么看`

---

### /adk-readiness

扫描仓库的 AI Coding 就绪度，输出多维度成熟度报告与改进建议。

**对应 skill**：`adk-readiness`

**用法**：`/adk-readiness [可选参数说明]`

**典型输入**：
- `/adk-readiness`
- `/adk-readiness only context and testing`
- `/adk-readiness compare with last run`
- `/adk-readiness target L3`

**适用场景**：
- 刚接手一个仓库，想知道是否适合 AI 开发。
- 想补齐文档、测试、规范、CI 等短板。
- 想量化改造前后的 readiness 提升。

---

### /adk-commit

提交当前工作区改动，自动生成规范化 commit message，并附加 TTADK Co-authored-by 签名。

**用法**：`/adk-commit`

**重要说明**：
- 推荐使用 `/adk-commit` 以确保 AI 代码贡献率被统计。
- 如果存在远端，会尝试 push；如果 push 失败，会保留本地 commit 并提示用户手动处理。
- 不会在 commit 流程中修改代码。

---

## SDD 命令

### /adk-sdd-brainstorm

对需求、ERD、资料链接或一个简单想法进行头脑风暴，帮助用户梳理方案并形成结构化思路文档。

**用法**：`/adk-sdd-brainstorm <想法、资料或链接>`

**适用场景**：
- 需求还很模糊，需要先拆模块和讨论方案。
- 已有 ERD / 学习资料，希望先沉淀成技术思路。
- 想先比较 2 到 3 套方案，再进入正式 spec。

---

### /adk-sdd-ff

Fast Forward 模式，一步生成 `spec.md`、`plan.md`、`tasks.md` 三个核心制品。

**对应 skill**：`adk-sdd-ff`

**用法**：`/adk-sdd-ff <功能描述或飞书文档链接>`

**产出**：
- `spec.md`
- `plan.md`
- `tasks.md`

**适用场景**：
- 需求边界清楚，希望快速推进。
- 已经过 brainstorm，方案已经比较明确。
- 需要用最短路径进入实现阶段。

---

### /adk-sdd-constitution

查看或更新项目宪章，定义项目级原则、编码标准和质量门禁。

**用法**：`/adk-sdd-constitution [内容或飞书文档链接]`

**示例**：
- `/adk-sdd-constitution`
- `/adk-sdd-constitution 新增规则：所有 API 端点必须有限流`
- `/adk-sdd-constitution https://lark-doc-with-coding-standards`

**产出**：`docs/CONSTITUTION.md`

---

### /adk-sdd-specify

基于描述或飞书文档创建功能规格。

**用法**：`/adk-sdd-specify <功能描述或飞书文档链接>`

**示例**：
- `/adk-sdd-specify 构建一个支持邮件和站内信渠道的用户通知系统`
- `/adk-sdd-specify https://lark-doc-link-to-prd`

**前置条件**：通常无，是功能规格阶段的第一个核心命令。

**产出**：`specs/YYYYMMDD-feature-name/spec.md`

---

### /adk-sdd-clarify

通过交互式问答完善规格，并同步所有已存在的下游制品。

**用法**：`/adk-sdd-clarify [可选的聚焦领域或反馈]`

**示例**：
- `/adk-sdd-clarify`
- `/adk-sdd-clarify 聚焦安全性需求`

**前置条件**：`spec.md` 必须存在。

**行为**：
- 每次提出最多 5 个针对性问题逐一交互。
- 每个回答被接受后更新 `spec.md`。
- 如果 `plan.md`、`technical-design.md`、`tasks.md` 已存在，也会同步更新。

---

### /adk-sdd-plan

创建包含实现计划、调研结论、数据模型和接口契约的规划制品。

**用法**：`/adk-sdd-plan [可选的约束条件]`

**前置条件**：`spec.md` 必须存在。

**产出**：
- `plan.md`
- `research.md`
- `data-model.md`
- `contracts/`
- `quickstart.md`

---

### /adk-sdd-erd

生成技术设计文档，适合需要图表、分层说明或复杂交互说明的功能。

**用法**：`/adk-sdd-erd [可选的聚焦领域]`

**前置条件**：通常需要 `spec.md`，复杂场景建议已有 `plan.md`。

**产出**：`technical-design.md`

**提示**：适合复杂功能、跨模块改造、团队评审前沉淀设计说明。

---

### /adk-sdd-tasks

将实现计划拆解为依赖有序的原子化任务。

**用法**：`/adk-sdd-tasks [可选的约束条件]`

**前置条件**：`spec.md` + `plan.md`

**产出**：`tasks.md`

**特性**：
- 任务包含 ID、依赖排序、文件路径。
- 支持并行标记 `[P]`。
- 面向实现执行，不是泛泛的 TODO 列表。

---

### /adk-sdd-analyze

对 `spec.md`、`plan.md`、`tasks.md` 做只读的交叉一致性和质量分析。

**用法**：`/adk-sdd-analyze [可选的聚焦领域]`

**前置条件**：`spec.md` + `plan.md` + `tasks.md`

**产出**：分析报告（输出到终端，不保存文件）。

**重点**：
- 发现规格遗漏、术语漂移、任务覆盖不足、与宪章冲突等问题。
- 这是制品质量检查，不是代码修改命令。

---

### /adk-sdd-implement

按照 `tasks.md` 的顺序执行实现任务并落地代码。

**用法**：`/adk-sdd-implement [可选：任务 ID、Phase 或反馈]`

**示例**：
- `/adk-sdd-implement`
- `/adk-sdd-implement Phase 2`
- `/adk-sdd-implement 修复支付模块的错误处理`

**前置条件**：`spec.md` + `plan.md` + `tasks.md`

---

### /adk-sdd-simplify

对刚完成的实现做收敛与简化，在不改变功能行为和任务边界的前提下减少重复、清理不必要复杂度并提升复用。

**用法**：`/adk-sdd-simplify [可选：任务 ID、文件路径、反馈]`

**示例**：
- `/adk-sdd-simplify`
- `/adk-sdd-simplify Task 2.3`
- `/adk-sdd-simplify src/payment/service.ts`

**前置条件**：已有最近完成的实现代码改动，通常在 `/adk-sdd-implement` 之后、`/adk-sdd-codereview` 之前执行

**重点**：
- 这是实现后的收敛命令，不负责新增功能，也不负责扩 scope。
- 默认关注重复逻辑、局部复杂度、已有抽象复用机会，以及不会改变行为的安全简化。
- 推荐链路：`/adk-sdd-implement -> /adk-sdd-simplify -> /adk-sdd-codereview -> /adk-commit`

---

### /adk-sdd-codereview

对实现代码变更做结构化代码审查，适合在 commit / Bits-Code MR 前使用。

**用法**：`/adk-sdd-codereview [可选：base:<ref>、Bits-Code MR/分支目标、mode:report-only|mode:autofix|mode:headless]`

**示例**：
- `/adk-sdd-codereview`
- `/adk-sdd-codereview base:origin/main`
- `/adk-sdd-codereview mode:report-only`

**前置条件**：已有实现代码改动，通常在 `/adk-sdd-implement` 之后执行

**重点**：
- 这是代码审查命令，不是 `spec.md` / `plan.md` / `tasks.md` 的制品分析命令。
- 默认关注正确性、测试、可维护性、规范对齐，并按 diff 自动补充安全、性能、可靠性等视角。
- 审查通过后再执行 `/adk-commit` 更合适。

---

### /adk-sdd-compound

扫描 `services/*` 代码、`specs/` 需求目录，以及可选的业务初始化材料，提取 15 个维度的可复用资产并沉淀到 `docs/` 目录。

**用法**：`/adk-sdd-compound [业务文档链接｜本地文档路径｜补充说明]`

**输入方式**：
| 输入 | 说明 |
|------|------|
| （无输入） | 自动扫描仓库中的 `services/*` 和 `specs/`，并由命令内部判断扫描模式 |
| Lark / Feishu 链接 | 先导出文档内容，再将其作为初始化材料参与 compound |
| 本地文档路径 | 读取 markdown/text 文档并作为初始化材料参与 compound |
| 纯文本说明 | 将补充背景、术语、业务规则作为初始化材料参与 compound |

**示例**：
- `/adk-sdd-compound` — 自动扫描当前仓库并沉淀知识资产
- `/adk-sdd-compound https://bytedance.larkoffice.com/wiki/xxx` — 从飞书文档初始化并参与知识提炼
- `/adk-sdd-compound /absolute/path/to/requirement.md` — 从本地 markdown/text 文档初始化
- `/adk-sdd-compound 这是业务背景和术语说明` — 用纯文本补充业务上下文


**扫描维度**（15 个）：
| 维度 | 输出位置 |
|------|---------|
| 设计模式 | `docs/arch/patterns.md` |
| 代码规范 | `docs/CODING.md` |
| 常见 Bug 与修复 | `docs/references/fixes.md` |
| 配置模板 | `docs/references/config-templates.md` |
| 测试用例模式 | `docs/references/testing-patterns.md` |
| 部署脚本 | `docs/references/deployment.md` |
| 可靠性实践 | `docs/RELIABILITY.md` |
| 安全实践 | `docs/SECURITY.md` |
| 架构索引 | `docs/arch/index.md` |
| 产品需求索引 | `docs/product-specs/index.md` |
| 参考资料索引 | `docs/references/index.md` |
| 项目宪章 | `docs/CONSTITUTION.md` |
| 质量标准 | `docs/QUALITY.md` |
| 需求模式 | `docs/product-specs/patterns.md` |
| 服务拓扑 | `docs/arch/service-topology.md` |

**SDD 工作流联动**：
compound 生成的知识文档会被以下 SDD 命令自动引用：
- `/adk-sdd-brainstorm` → `docs/arch/patterns.md`、`docs/references/fixes.md`
- `/adk-sdd-specify` → `docs/product-specs/patterns.md`
- `/adk-sdd-plan` → `docs/arch/service-topology.md`、`docs/arch/patterns.md`
- `/adk-sdd-implement` → `docs/CODING.md`、`docs/references/config-templates.md`

**适用场景**：
- 项目初始化后，想快速沉淀已有代码中的知识资产。
- 业务方先给了飞书文档或普通需求文档，希望直接辅助初始化。
- 大型重构前，想了解当前架构和代码模式。
- 定期维护 docs/ 中的知识资产，保持与代码和业务材料同步。

---

### /adk-sdd-archive

归档历史功能制品，维护归档索引并压缩旧功能目录。

**用法**：
- `/adk-sdd-archive`
- `/adk-sdd-archive --dry-run`
- `/adk-sdd-archive --json`

**适用场景**：
- 功能已经完成，想精简 `specs/` 目录。
- 想保留归档记录，但减少活跃制品干扰。

## TTADK CLI 命令

### `ttadk init [project-name]`

初始化 TTADK 项目，交互式选择 AI 工具、语言和 Preset。

| 选项 | 说明 | 示例 |
|------|------|------|
| `-t, --ai-tool <tool>` | 指定 AI 工具 | `ttadk init -t claude` |
| `-l, --language <lang>` | 语言偏好（`en`/`zh`） | `ttadk init -l zh` |
| `-p, --preset <preset>` | 指定 Preset | `ttadk init -p ttadk/frontend` |
| `--no-git` | 跳过 Git 仓库初始化 | `ttadk init --no-git` |
| `-H, --here` | 在当前目录初始化 | `ttadk init -H` |

**可用 Preset**：

| Preset | 说明 |
|--------|------|
| `ttadk/common` | 通用插件集合 |
| `ttadk/frontend` | 前端开发（Lynx + 平台） |
| `ttadk/backend` | 后端开发（GDP + Kitex + Hertz） |
| `ttadk/lynx` | Lynx 开发专用 |
| `empty` | 不安装任何插件 |

Preset 支持指定分支：`ttadk init -p ttadk/frontend@dev`

---

### `ttadk code`

启动配置的 AI 编程工具，含 SSO 认证、模型发现和 API 路由。

| 选项 | 说明 | 示例 |
|------|------|------|
| `-t, --tool <tool>` | 覆盖 AI 工具 | `ttadk code -t claude` |
| `-m, --model <model>` | 指定模型 | `ttadk code -m claude-sonnet-4-5` |
| `-a, --args <args>` | 传递额外参数给 AI 工具 | `ttadk code -a "--verbose"` |
| `-p, --proxy <mode>` | 代理模式：`s`（服务端）/ `c`（客户端） | `ttadk code -p s` |

---

### `ttadk handoff`

TTADK 的异步 handoff CLI，用于提交、查看、继续、停止和同步异步任务。

**典型子命令**：
- `ttadk handoff submit`
- `ttadk handoff list`
- `ttadk handoff detail`
- `ttadk handoff stop`
- `ttadk handoff continue`
- `ttadk handoff sync`

---

### `ttadk plugin <subcommand>`

| 子命令 | 说明 | 示例 |
|--------|------|------|
| `install <name>` | 安装插件 | `ttadk plugin install my-plugin` |
| `uninstall <name>` | 卸载插件 | `ttadk plugin uninstall my-plugin` |
| `list` | 列出已安装插件 | `ttadk plugin list` |
| `info <name>` | 查看插件详情 | `ttadk plugin info my-plugin` |
| `update [name]` | 更新插件 | `ttadk plugin update --all` |

---

### `ttadk sync`

同步配置变更。手动修改 `.ttadk/config.json` 后执行。

| 选项 | 说明 |
|------|------|
| `-d, --dry-run` | 预览变更，不实际执行 |
| `-F, --force` | 跳过确认 |

---

### `ttadk config`

交互式 MCP 参数配置。扫描 MCP 配置文件中未配置的 `{{KEY}}` 模板变量。

---

### `ttadk upgrade`

升级 TTADK 到最新版本。

| 选项 | 说明 |
|------|------|
| `-f, --force` | 强制重新安装 |
| `-r, --registry <url>` | 自定义 npm registry |
| `--pm <pm>` | 指定包管理器（npm/pnpm/bun/yarn/auto） |

---

### `ttadk skills read <skill-name>`

读取并显示指定 Skill 的内容。

## 常用工作流模式

### 基础公共 Skill / Command 对照

| 命令 / 能力 | 对应 skill | 说明 |
|------------|------------|------|
| `/adk-readiness` | `adk-readiness` | 评估仓库 readiness |
| `/adk-help` | `adk-help` | 回答 TTADK、SDD 工作流与当前下一步 |
| `ttadk handoff *` | 无独立 skill | 通过命令管理 handoff 异步任务 |

### SDD Skill / Command 对照

| 命令 | 对应 skill | 产出/作用 |
|------|------------|-----------|
| `/adk-sdd-brainstorm` | `adk-sdd-brainstorm` | 头脑风暴与方案梳理 |
| `/adk-sdd-ff` | `adk-sdd-ff` | 一步生成 `spec.md`、`plan.md`、`tasks.md` |
| `/adk-sdd-constitution` | `adk-sdd-constitution` | 维护项目原则 |
| `/adk-sdd-specify` | `adk-sdd-specify` | 创建 `spec.md` |
| `/adk-sdd-clarify` | `adk-sdd-clarify` | 澄清并同步设计制品 |
| `/adk-sdd-plan` | `adk-sdd-plan` | 创建实现计划与设计骨架 |
| `/adk-sdd-erd` | `adk-sdd-erd` | 生成技术设计文档 |
| `/adk-sdd-tasks` | `adk-sdd-tasks` | 拆解实现任务 |
| `/adk-sdd-analyze` | `adk-sdd-analyze` | 只读分析制品质量 |
| `/adk-sdd-implement` | `adk-sdd-implement` | 执行任务并实现代码 |
| `/adk-sdd-simplify` | `adk-sdd-simplify` | 对刚完成的实现做收敛、去重和复杂度清理 |
| `/adk-sdd-codereview` | `adk-sdd-codereview` | 对实现代码执行结构化审查 |
| `/adk-commit` | `adk-commit` | 提交当前改动 |
| `/adk-sdd-archive` | `adk-sdd-archive` | 归档历史制品 |

### 完整功能开发

```bash
ttadk init my-project -t claude -p ttadk/backend
cd my-project && ttadk config && ttadk code

/init
/adk-readiness
/adk-sdd-constitution
/adk-sdd-brainstorm "想法或资料"      # 可选
/adk-sdd-specify "功能描述"
/adk-sdd-clarify
/adk-sdd-plan
/adk-sdd-erd                         # 可选
/adk-sdd-tasks
/adk-sdd-analyze                     # 可选
/adk-sdd-implement
/adk-sdd-simplify                    # 可选，推荐在 codereview 前执行
/adk-sdd-codereview                      # 可选，推荐在 commit 前执行
/adk-commit
/adk-sdd-archive
```

### 快速推进

```bash
/adk-readiness
/adk-sdd-brainstorm "构建一个数据仪表板"   # 可选
/adk-sdd-ff "构建一个数据仪表板"
/adk-sdd-clarify "增加日期范围筛选"        # 如需修正
/adk-sdd-implement
/adk-sdd-simplify
/adk-sdd-codereview
/adk-commit
```

### 异步 Handoff 协作

```bash
ttadk handoff submit "基于最新的 Spec 继续实现并产出代码"
ttadk handoff list
ttadk handoff sync -t <task_id>
```

## 插件开发

详细文档参考：[TTADK - 插件自定义系统](https://bytedance.larkoffice.com/wiki/HGCMwBOxci0nddkLwybcu5cMnTd)

官方插件市场仓库：https://code.byted.org/tiktok/ttadk_plugin_market

### 插件结构

```text
plugins/{namespace}/{plugin_name}/
├── .ttadk-plugin/
│   └── plugin.json              # 插件清单（必需）
├── commands/                    # Slash 命令（Markdown 格式）
├── mcps/                        # MCP 服务配置（JSON 格式）
├── skills/                      # 知识模块（SKILL.md 格式）
├── agents/                      # 子代理定义
├── memory/                      # 项目记忆内容
└── resources/                   # 脚本、模板等资源
```

### 插件清单（plugin.json）

```json
{
  "name": "namespace/plugin_name",
  "display_name": "My Plugin",
  "version": "1.0.0",
  "description": "插件描述",
  "author": {"name": "Author", "email": "author@bytedance.com"},
  "tags": ["tag1", "tag2"],
  "commands": [{"name": "my-cmd", "path": "commands/my-cmd.md"}],
  "mcps": [{"name": "my-mcp", "path": "mcps/my-mcp.json"}],
  "skills": [{"name": "my-skill", "path": "skills/my-skill"}],
  "agents": [{"name": "my-agent", "path": "agents/my-agent.md"}]
}
```

### Preset 配置（preset.json）

```json
{
  "name": "namespace/preset_name",
  "description": "Preset 描述",
  "plugins": ["namespace/plugin1", "namespace/plugin2"]
}
```

### 开发流程

1. 克隆插件市场仓库，新建分支。
2. 在 `plugins/{namespace}/{plugin_name}/` 下创建插件目录。
3. 创建 `.ttadk-plugin/plugin.json`。
4. 按需添加 commands / mcps / skills / agents。
5. 本地测试：`ttadk plugin install {namespace}/{plugin_name} -b feat/xxx`。
6. 测试通过后合码。

### 扩展官方命令

要在官方 `adk` 命名空间下新增命令，创建插件并添加对应的命令文件。对于工作流命令，建议按当前分组方式组织，例如：

- `commands/adk/my-command.md`
- `commands/adk/sdd/my-step.md`

安装后会与官方命令并列。名称冲突时后安装覆盖先安装。

### 命名规范

- Namespace 和插件名：仅允许小写字母、数字和下划线。
- Preset 名：仅允许小写字母、数字和下划线。
- 组件的 `compatible_tools` 字段可限制该组件安装到哪些 AI 工具。
