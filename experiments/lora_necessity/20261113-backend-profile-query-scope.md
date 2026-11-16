# Backend profile 元数据查询：当前不执行

日期：2026-11-13，Asia/Shanghai。

状态：**NOT_EXECUTED_AUTHORIZATION_MISSING**。
同时存在 `MISSING_IMAGE_ID` 和 `MISSING_LIFECYCLE_PROVIDER`。
本文件记录下一次查询尚未满足的条件，不是可运行查询或新的连接许可。

## 目的与目标

只为已批准的专属 tokenizer 后端设计收集准确 runtime 和生命周期元数据。
既有目标为 `rl-server`，账号 `sitian`，预期 hostname `n232-195-203`、
UID 1054。准确地址和跳板绑定在既有成功查询的私有 `scope.json`；
本轮只读取该记录，不重新解析路由、检查认证或连接目标。
2026-11-13 16:09:15 +08:00 的成功观察不代表当前连接或权限复验。

## 选择器及来源

| 选择器 | 已有依据 | 本次可用范围 |
| --- | --- | --- |
| 目标身份与 Docker endpoint | 成功 capability 查询的 scope、receipt、stdout 及 verification | 本地核查；原单次远端许可已使用 |
| 镜像准确 ID/摘要 | 11 月 5 日记录没有选定镜像；本次核查的任务记录也没有提供候选 ID | 缺失；不扫描其他镜像或工作负载 |
| 历史解释器 | 父 admission 规格 §2 的准确路径，版本要求为 Python 3.12.13 | 定位线索，不作为镜像；未授权执行或挂载 |
| 依赖元数据路径 | transformers 要求 4.57.1，其余实际依赖精确清单尚未冻结 | 缺失；不猜 site-packages 路径，不导入库 |
| 生命周期提供方 | 历史 POSIX supervisor、comparison remote helper 和部署指南 | 仅供代码参考；不是已准入的容器租期服务 |

## 准确命令与允许返回字段

**准确远端 command/argv：未形成，因此没有可执行命令。**
没有生成带占位符的 `docker inspect`、平台命令或操作 helper。
只有取得真实、获准的镜像与控制接口选择器后，才冻结一次完整命令并另行审阅。

届时镜像查询只能返回指定镜像的 ID/digests、OS/architecture、声明的
user/entrypoint/command、volumes 及明确列出的 runtime 需求；不得返回
环境变量、history、layers、其他镜像或容器。读取入口不授权运行入口。
若需要宿主依赖元数据，必须另外列出准确路径、读取命令及字段；
不自动执行解释器，不 import transformers/tokenizers，不读取模型文件。
生命周期字段须由实际平台接口决定；现在不编造 API 名称或响应字段。

## 后续查询的限制

若完整查询另获批准：一次连接，本地工作至多 55 秒，总窗 60 秒；
远端命令至多 35 秒加最多 2 秒终止余量；stdout/stderr 各 1,048,576 字节。
严格主机密钥检查，不委托凭据、不转发 agent、不复用连接、不交互回退、
不自动重试、不替换目标。任何缺失、身份变化、字段错误或超限均停止。

本轮本地 profile 证据目录为
`/tmp/tlf-tokenizer-backend-profile-xf6M1bsp`，目录 0700、文件 0600。
后续获准查询必须新建独占私有目录，不能复用本目录或旧查询目录。
旧 capability helper 仅供构造参考，不在封存目录中执行或修改。

## 禁止操作及授权状态

本轮仅执行本地文档、代码和既有证据核查。没有 SSH、镜像枚举、
pull/build、安装、资源配置、创建/启动容器、私有数据传输、
tokenizer/模型加载、GPU、训练、提交或推送。

用户“继续”授权执行第一阶段本地计划，不替代准确查询、行为验证、
真实审计各自的许可。没有新的远端 query receipt 或 stdout/stderr；
不为未执行的查询制造空回执。无远端动作状态写入 `profile.json`。

下一责任方：用户或目标平台维护者提供可用于此任务的已有镜像准确
ID/摘要，以及获准的生命周期接口/服务资料；随后由 agent 冻结准确
只读查询，交用户批准。若没有现成镜像或服务，另议准备范围，
不把提供资料的请求变成默认允许安装或自建服务。
