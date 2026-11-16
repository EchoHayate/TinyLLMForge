# 旧远端运行环境核查结果

2026-11-13。状态：`EXISTING_RUNTIME_METADATA_OBSERVED_NOT_AUDIT_ADMITTED`。

**本次确实连上了原服务器，旧 Python 环境存在且能执行本次 stdlib
查询，关键版本与原要求一致。不是环境丢失，也不是这次登录失败。**

## 实测结果

- 目标报告时间：2026-11-13 19:16:21 +08:00。
- 跳板与目标机均通过 GSSAPI 认证；hostname/UID 与原目标匹配。
- 一次连接，SSH exit 0，耗时 2.079 秒；身份与查询完成标记齐全。
- stdout 2,512 字节、stderr 27,693 字节，均未截断、未超限。
- 唯一发现的元数据目录属于已指定旧环境，匹配 Python 3.12 布局。
- 读取 75 个安装包的名称/版本，四个必需发行包均存在。

| 项目 | 本次观察 | 对原要求 |
| --- | --- | --- |
| Python | 3.12.13 | 匹配固定版本 |
| transformers | 4.57.1 | 匹配固定版本 |
| tokenizers | 0.22.2 | 原来未冻结；现取得安装元数据 |
| huggingface-hub | 0.36.2 | 原来未冻结；现取得安装元数据 |
| jinja2 | 3.1.6 | 原来未冻结；现取得安装元数据 |

后三项不是通过与原有精确版本比较而“验收通过”，而是本次新观察。
完整版本表和具体路径保存在私有 `observed-runtime.json` 中，
与原始 stdout/receipt 摘要绑定。此表不是已获准的加载 profile。

## 执行边界

用户批准了 `20261113-existing-runtime-query-scope.md` 中的准确单次查询。
范围文档保留其执行前状态，本结果记录后续获准和执行，不覆盖旧文档。
远端 command 逐字取自该文档，SSH 参数沿用既有成功查询的绑定；
不自动重试、不放宽认证或主机密钥检查。

用指定解释器的 `-I -S -B` 运行 stdlib 查询，禁普通 site 启动处理，
只在已指定 env 中发现和读取安装元数据。没有导入 transformers、
tokenizers 或 torch，没有读取模型/答案，没有加载 tokenizer、模型、
启动训练、使用 GPU、安装依赖、创建容器或远端文件。

本次证明“解释器能执行该查询、安装元数据存在”，不证明第三方库能正常
导入或实际 tokenizer 加载成功。普通 site 启动及完整运行路径等价性
仍未验证，不能把 `-S` 下的元数据表直接当作真实加载身份。

## 对下一步的影响

- 不再把“找不到旧环境”或“没有依赖版本信息”作为当前事实。
- 自动断连清理已被用户取消，不因缺独立清理服务阻止推进。
- 内存/进程线程、双端磁盘/输出限制仍未落实为已验证的运行配置；
  本次未查资源能力，也不把旧 Docker 声明当作当前生效证据。
- 下一步以实测 runtime 信息处理资源配置与真实 tokenizer 检查的
  剩余条件；不因本次查询成功自动启动实验或切换无约束宿主运行。

## 本地检查与保全

执行前检查了准确 argv、shell/Python 语法；用四个短小本地 stdlib
子进程检查正常输出、stdout 超限、stderr 超限、超时与直接子进程回收。
这些只验证查询采集器，不是后端行为测试或模型实验结果。

私有证据目录：`/tmp/tlf-runtime-metadata-20261113-q73tiizg`。
目录 0700、文件 0600；每个生成文件独占创建。
`verification.json` 绑定 scope/helper、原始输出、观察表、报告与台账追加。
保全入口为 748 个文件；除允许追加的进度台账外不改原文件，
不修改 SSH 配置或 known_hosts，不提交、推送或更新记忆。

`resource_limits_enforced`、`remote_cleanup_verified`、`real_tokenizer_loaded`
及 `benchmark_eligible` 仍为 false。本次没有新增 LoRA 效果结论。
