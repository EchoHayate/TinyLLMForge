# TTADK 常见问题与排障

> 官方 FAQ 文档（提 oncall 前必看）：[TTADK FAQ](https://bytedance.larkoffice.com/wiki/RJsSwFVvhiD0DPkLgXicfoKknbh)

## 入门指引

### Q：刚接触 TTADK，第一步是什么？

1. 如果不了解 SDD 概念，参见 `sdd-workflow.md` 或阅读：[Spec-Driven Development - AI Coding 时代的研发效能](https://bytedance.larkoffice.com/wiki/LBZSwHJOPiXiOpkrMBCcwXS1nmg)
2. 如果不了解 TTADK，参考：[TTADK - TikTok AI-Driven Development Kit (中文)](https://bytedance.larkoffice.com/wiki/Gw0ewxEbHi1K0NkVd2YcNwvVnTg)、[TTADK (Spec Kit) 实践分享](https://bytedance.larkoffice.com/wiki/VC9MwaZDxiP7ljkgR8IcXPzpnkg)
3. 安装 TTADK：`npm install -g @byted/ttadk --registry=https://bnpm.byted.org`
4. 初始化项目：`ttadk init my-project`
5. 启动 AI 工具：`ttadk code`
6. 在 AI 工具中执行 `/init` 生成 `CLAUDE.md` / `AGENTS.md`，并人工审查后再使用
7. 先执行 `/adk-readiness` 了解仓库 readiness，再根据场景进入 `/adk-sdd-brainstorm`、`/adk-sdd-specify` 或 `/adk-sdd-ff`

### Q：如何用 TTADK 开始一个需求？

> Lynx 同学需参考 [Spark Skill Lynx 开发全流程验证](https://bytedance.larkoffice.com/wiki/U6LYw6OnhipwMNkRab8c943PnVe) 配置 Figma MCP Token

1. 先执行 `/adk-readiness`，确认仓库在上下文、测试、规范、CI 等方面是否足够 AI Friendly
2. 复杂需求建议先准备 ERD（技术方案）或直接执行 `/adk-sdd-brainstorm`，先拆分模块、确认边界和技术方向。ERD 模板参考：[前端技术方案模板](https://bytedance.larkoffice.com/wiki/IKbfwRA3aiCZGokZXD5cRdQgnZf)、[后端技术方案模板](https://bytedance.larkoffice.com/wiki/ZBuawQfg9isvG2k7gXbcNMOhnSg)、[TTADK Lynx ERD 模版](https://bytedance.larkoffice.com/wiki/BmBBwcpypi3RyQkTVrBcvQalnXI)
3. Clone 代码到本地（多仓库可新建临时目录统一 clone，更正式的方案参考 [基于 SDD 的代码仓库管理方案](https://bytedance.sg.larkoffice.com/docx/RjbrdkxmroH6uUxqSCEl5epxg1d)）
4. 在项目目录下执行 `ttadk init`
5. 执行 `ttadk code` 启动 AI 工具
6. 如果没有 `CLAUDE.md` / `AGENTS.md`，执行 `/init` 生成并审查
7. 选择路径：
   - 标准模式：`/adk-sdd-specify <需求描述或文档链接>` → `/adk-sdd-plan` → `/adk-sdd-tasks` → `/adk-sdd-implement` → `/adk-sdd-codereview`
   - 快速模式：`/adk-sdd-ff <需求描述或文档链接>` → `/adk-sdd-implement` → `/adk-sdd-codereview`

---

## SDD 工作流问题

### Q：生成内容不符合预期，如何排查？

1. 先审查 spec / plan / tasks 是否有遗漏。优先执行 `/adk-sdd-clarify` 修正；如果已经有完整三件套，也可以先跑 `/adk-sdd-analyze` 看覆盖和一致性问题
2. 检查 ERD 质量，确认是否有关键信息漏掉
3. 检查 `CLAUDE.md` / `AGENTS.md`，补充仓库上下文（术语、架构设计等）。没有的话先执行 `/init`
4. 检查代码库质量，确认分层是否清晰、抽象是否合理
5. 检查 Skills，确认需要的技术栈 skill 是否已安装。缺失的在 TTADK 群反馈
6. 最后在群里寻求支持。关键提示：精准管理上下文，命令之间使用 `/clear`，TTADK 命令都是可重入的

### Q：可以跳过工作流中的某些阶段吗？

必须的依赖关系：
- 标准模式：`/adk-sdd-specify` → `/adk-sdd-plan` → `/adk-sdd-tasks` → `/adk-sdd-implement`
- 快速模式：`/adk-sdd-ff` → `/adk-sdd-implement`

可跳过或按需使用的阶段：`/adk-sdd-brainstorm`、`/adk-sdd-clarify`、`/adk-sdd-erd`、`/adk-sdd-analyze`、`/adk-sdd-codereview`

### Q：`/adk-sdd-clarify` 可以执行多次吗？

可以。每次提出最多 5 个新问题并更新所有已存在的制品，可按需多次执行。

### Q：手动修改了 spec.md 会怎样？

手动修改没问题，但下游制品不会自动更新。需执行 `/adk-sdd-clarify` 或重新运行下游命令来同步。

### Q：可以结合 Vibe Coding 进行小改动吗？

可以。SDD 更适合结构化功能开发，小修小补可直接编辑或 Vibe Coding。若已进入 SDD 流程，可用 `/adk-sdd-implement [反馈]` 修正实现问题；如果根因是规格不清晰，先执行 `/adk-sdd-clarify`。

### Q：如何推倒重来？

删除或重命名 `specs/` 下的功能目录，然后重新执行 `/adk-sdd-specify` 或 `/adk-sdd-ff`。

### Q：`/adk-sdd-analyze` 是做什么的？

它是只读制品分析命令，主要检查 `spec.md`、`plan.md`、`tasks.md` 三者的一致性、覆盖度和质量，不负责改代码，也不是代码审查命令。

### Q：`/adk-sdd-codereview` 是做什么的？

它是代码审查命令，主要检查实现代码 diff 的正确性、测试覆盖、可维护性、规范对齐，以及按需触发的安全、性能、可靠性等风险。

和 `/adk-sdd-analyze` 的区别：
- `/adk-sdd-analyze` 看的是 `spec.md` / `plan.md` / `tasks.md` 这些制品
- `/adk-sdd-codereview` 看的是代码改动本身

推荐在 `/adk-sdd-implement` 之后、`/adk-commit` 之前执行。

### Q：什么时候用 `/adk-sdd-ff`，什么时候走标准模式？

适合用 `/adk-sdd-ff` 的情况：
- 需求已经比较清楚
- 想快速拿到 `spec.md`、`plan.md`、`tasks.md`
- 团队已经对方案基本达成一致

适合走标准模式的情况：
- 需求仍然模糊
- 需要先拆方案或做更多澄清
- 改动较大，想逐阶段审查产物质量

---

## Readiness 与异步 Handoff

### Q：怎么知道当前仓库适不适合直接开始 AI 开发？

执行 `/adk-readiness`。它会从上下文工程、文档、测试、代码组织、安全治理、版本协作、SDD readiness 等多个维度给出成熟度报告和改进建议。

### Q：什么时候该先跑 `/adk-readiness`？

推荐在以下情况先跑：
1. 新接手仓库
2. 仓库历史包袱重、上下文不清晰
3. 想引入 TTADK / AI Coding，但不确定基础设施是否准备好
4. 想量化改造前后的 readiness 提升

### Q：怎么把任务放到异步环境执行？

使用 `ttadk handoff`：

- 查看状态：`ttadk handoff list`
- 发起异步任务：`ttadk handoff submit "基于最新的 Spec 发起一个异步 SDD 任务"`
- 继续失败任务：`ttadk handoff continue "修复 type errors" -t <task_id>`
- 同步结果：`ttadk handoff sync -t <task_id>`

### Q：什么时候适合用异步 handoff？

适合以下场景：
- 本地不方便长时间执行
- 需要把一轮 SDD / 实现放到云端异步跑
- 想先离线等待结果，再把分支同步回来审查

### Q：`/adk-readiness` 和 `ttadk handoff` 对应什么 skill？

- `/adk-readiness` 对应 `adk-readiness`
- `ttadk handoff *` 通过 `/adk/handoff.md` 命令入口管理，无独立 skill

---

## 配置与版本管理

### Q：哪些文件应该提交到 Git？

| 路径 | 是否提交 | 说明 |
|------|---------|------|
| `.ttadk/` | 是 | 团队共享配置 |
| `docs/CONSTITUTION.md` | 是 | 团队共同维护的项目原则 |
| `CLAUDE.md` / `AGENTS.md` | 是 | AI 工具规范，由 `/init` 生成 |
| `specs/` | 建议 | 定期用 `/adk-sdd-archive` 归档 |
| `specs/doc_export/` | 否 | 加入 `.gitignore`（中间产物） |
| `.claude/` / `.cursor/` 等 | 建议 | 插件安装的配置，可在团队间共享 |

### Q：Constitution 是什么？

`docs/CONSTITUTION.md` 定义项目级原则（编码标准、架构决策、质量门禁）。所有 SDD 命令执行前都会读取。通过 `/adk-sdd-constitution` 编辑。每个项目一份，在仓库根目录。

### Q：可以同时开发多个功能吗？

可以。每个功能在 `specs/` 下有独立目录。TTADK 自动检测最新的功能目录。无需手动切换，`/clear` 后下一个命令会自动拾取最新的 `specs/` 目录。

### Q：如何设置语言偏好？

编辑 `.ttadk/config.json`，设置 `"preferred_language": "zh"`（中文）或 `"en"`（英文）。

---

## AI 代码贡献率追踪

### Q：如何确保 AI 代码贡献被统计？

两种方式：
1. **使用 `/adk-commit`**（推荐），自动添加追踪签名
2. **通过 `ttadk code` 启动**并选择模型，走 TTADK 模型网关

统计时机为 commit 代码被 MR 合入，而不仅仅是 commit。更新频率约 T+2。

> 注意：必须在项目仓库目录下启动 `ttadk code`，否则无法获取 repo 信息导致统计不到。

### Q：Squash merge 影响统计吗？

- Codebase squash merge：不影响
- 手动 squash merge：保留 `co-authored-by:ttadk` 签名即不影响

### Q：在哪里查看我的 AI 贡献数据？

- 个人数据看板：devmind.bytedance.net
- 参考文档：[字节跳动 标准代码行度量 说明](https://bytedance.larkoffice.com/wiki/wikcnoLjTWTkdSeToWMWyxLwHcc)、[TT Eng AI 贡献率 自助排查工具和文档](https://bytedance.larkoffice.com/wiki/NG7cwWKGWiBa2Ekg14UcjShOnUh)

---

## 安装与环境问题

### Q：`ttadk init` 报 Git 权限错误

```text
Error: git@code.byted.org: Permission denied (publickey,gssapi-with-mic)
```

**解决方案**：
1. 确认有插件市场仓库的权限：https://code.byted.org/tiktok/ttadk_plugin_market
2. 执行 `kinit`（如 `kinit your.email@bytedance.com`）后重试 `ttadk init`
3. 确保 TTADK 不是用 `sudo` 安装的。如果是，卸载后不用 `sudo` 重装

### Q：DevBox 安装，npm 全局路径问题

DevBox 默认 npm 全局路径在 `/usr/local/` 下（属于 root），需修改：

```bash
# 添加到 ~/.bashrc
export NPM_CONFIG_PREFIX="$HOME/.npm-global"
export PATH="$HOME/.npm-global/bin:$PATH"
source ~/.bashrc
```

然后不用 `sudo` 重新安装 TTADK。

### Q：证书错误：SELF_SIGNED_CERT_IN_CHAIN

- **安装阶段报错**：先验证网络权限 https://bnpm.byted.org/@byted/ttadk，不通则执行 `npm config set cafile /etc/ssl/cert.pem`
- **`ttadk code` 启动阶段报错**：
```bash
echo 'export NODE_OPTIONS=--use-bundled-ca' >> ~/.zshrc
source ~/.zshrc
ttadk code
```

### Q：`ttadk code` 找不到 AI 工具命令

AI 工具的 CLI 未安装或不在 PATH 中。

**解决方案**：安装工具的 CLI，或编辑 `~/.ttadk/setting.json` → `ai_tool_commands` → 设置工具的完整路径。
