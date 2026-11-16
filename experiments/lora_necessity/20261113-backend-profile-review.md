# Backend profile 核查结果

日期：2026-11-13，Asia/Shanghai。状态：**PROFILE_INCOMPLETE**。
本地调查与缺项报告完成；后端未准入，不是实现或实验成功。

## 1. 可用事实

- 本轮 checkout 为 `experiments/feedback-active-verification`，
  HEAD `93b4e8e0dc5174db0426a5da9e29cc360ed331a7`；未切换分支。
- 既有原始输出记录了 2026-11-13 16:09:15 +08:00 的目标身份匹配、
  Docker 24.0.9/cgroupfs/v1/overlay2 和资源能力声明。
  所选用户 cgroup 目录不可写；这些都是旧查询事实，不是当前配置验收。
- 父规格要求 Python 3.12.13、transformers 4.57.1。旧宿主解释器路径
  是定位线索，不是已验证容器 runtime。其余依赖须按父规格另行盘点。
- 本轮完整读取 `selection_model_supervisor.py`，并核查 comparison
  supervisor、remote helper 的启动路径和部署指南。现有实现并非
  “完全没有远端监督”：remote helper 写入绑定 run ID、源码和
  PID/start_ticks 的启动记录，忽略 SIGHUP 后 exec supervisor；
  supervisor 创建 worker 独立进程组，执行 TERM/KILL、等待和回执。
- 可复用的是身份绑定、单次启动、失败保留和终止/回收思路。
  **不能直接复用为新后端验收**：监督期限从 `supervise()` 入口起算，
  默认 595+5 秒；旧部署总窗 1200 秒，bridge 588+12 秒。
  它没有本次容器租期及资源/存储配置，不能证明 SSH 断开后的全部资源释放，
  也不能把进程组消失当作任意后代和容器/空间域均已释放。

## 2. 缺失事实

| 缺项 | 需要的具体证据 |
| --- | --- |
| `MISSING_IMAGE_ID` | 获准的已有镜像准确 ID/摘要及架构、入口、用户、volumes 元数据 |
| `MISSING_RUNTIME_INVENTORY` | Python/transformers 和 tokenizers、huggingface-hub、jinja2 等实际加载依赖的精确清单与路径 |
| `MISSING_LIFECYCLE_PROVIDER` | 实际控制接口、可控制对象、创建失败/失联处理、独立到期及清理契约与使用权限 |
| `MISSING_RESOURCE_ALLOCATION` | 准备、审计、采集、远端控制及新增辅助任务的归属与硬限额分配 |
| `MISSING_STORAGE_COVERAGE` | 双端全部新文件和副本、辅助挂载、元数据/inode 的计量与硬上限 |
| `MISSING_DEADLINE_MAPPING` | 初次连接起 540/600 秒预算的保守跨时钟映射和清理余量 |
| `MISSING_EXACT_QUERY_AUTHORIZATION` | 真实选择器下完整只读命令及单次执行许可 |
| `MISSING_BEHAVIOR_AND_AUDIT_ADMISSION` | 后续独立行为验证、runtime/输入身份和私有传输分别获准 |

这些是本次证据不足，不代表全机没有可用镜像或生命周期服务。
没有扩大搜索到其他工作负载，没有因历史 supervisor 存在而取消硬关口。

## 3. 拟定的准确配置

**尚不能拟定可执行配置。** 镜像、控制 API、实际资源组、挂载路径、
双端额度及期限映射未确定，不提供伪造的 `docker run` 或默认参数串。
固定的只是已批准约束：聚合 4,294,967,296 字节、最多 8 个进程/线程，
全部新文件峰值 134,217,728 字节、结果 67,108,864 字节，
stdout/stderr 各 1,048,576 字节；540 秒工作、600 秒总窗、
清理最多 60 秒且在原总窗内。无 swap 扩容、单线程配置、禁 core dump。

存储核算必须同时覆盖：远端新快照/输入、结果、临时文件、辅助挂载、
控制文件与日志；本地新增快照/输入副本、接收结果、捕获流和证据；
容器可写层、隐式 volume、`/dev/shm`、缓存和 runtime 辅助写入
须明确禁写或纳入额度。已有只读源与本次新副本分开计量。
现在没有具体路径/限额覆盖证明，因此也不能保证这一清单在镜像选定后无需补充。

若双端同时保留各一份达到上限的结果，仅这两份逻辑内容就达到 128 MiB，
不给其他新文件留余量；不能因此宣称整体峰值预算已满足。
优先设计不同时落两份完整结果的传输方式，但本轮没有实现或验收该方式。
额度复用须以实际释放为前提，不把累计写入量当峰值占用；
逻辑字节、分配块和元数据/inode 要分别说明约束。

资源核算需列出新增远端控制程序、SSH 会话辅助任务、准备/写入/采集程序、
审计 worker、init 和线程；所有并存硬限额之和必须不超总预算。
既有 daemon/SSH 服务的控制面边界需要平台认可，不能隐含排除新增 helper。
本地控制、传输和接收任务也须明确记录归属；尚无完整 allocation。

## 4. B01–B10 覆盖

| 行 | 本轮证据及结论 |
| --- | --- |
| B01 runtime | 只有版本要求及历史路径；缺镜像和完整依赖，未通过 |
| B02 内存 | Docker 声明支持，不是聚合限额或行为证据，未通过 |
| B03 任务数 | 只有能力声明；所有 init/线程/helper 的合计尚未证明，未通过 |
| B04 存储 | 已列双端覆盖要求与双结果风险；无准确配额/路径配置，未通过 |
| B05 输出 | 既有本地输出组件不证明生产端和容器全链路上限，未通过 |
| B06 期限/断连 | 历史 supervisor 时钟范围不符；缺独立租期和时间映射，未通过 |
| B07 清理 | 历史 PID 身份与回执思路可参考；缺新对象/后代/端点/空间释放证据，未通过 |
| B08 输入 | 旧五文件指纹和投影限制保留；无本轮真实快照或传输许可，未通过 |
| B09 停止 | 本轮因缺选择器/许可不连接；运行时异常行为尚未验证，未通过 |
| B10 保全 | 本轮 baseline 哈希及台账前缀由最终 seal 复核；仅属本地保全，不是后端行为验收 |

## 5. 剩余授权

已批准书面设计和第一阶段本地执行；历史文档的 pending 标题保留原貌，
新进展追加台账，不回写旧封存。当前查询结论为
`NOT_EXECUTED_AUTHORIZATION_MISSING`，另有镜像/服务缺项。
没有新的连接、操作 helper 或远端查询回执。
准确元数据查询、有限行为探针、真实 tokenizer 和私有投影传输
仍是不同关口；本轮未运行组件测试，也未增加模型效果证据。

## 6. 下一责任方

用户/平台维护者先提供本任务可用的已有镜像 ID/摘要，以及独立到期清理
服务/接口的资料和使用范围；无需提供凭据内容或其他任务信息。
Agent 据此冻结窄范围只读查询，获准后单次获取真实元数据。
若平台无现成支持，需明确另行准备范围或调整方案，不以普通进程轮询兜底。
当前不进入依赖未知 API 的 launcher 实现，也不再叠加模拟准入组件。

机器可读 profile 与本轮验证位于
`/tmp/tlf-tokenizer-backend-profile-xf6M1bsp`。
这是本地自核查和可操作缺项报告，未标记独立审阅通过或
`BACKEND_PROFILE_REVIEWED_NOT_ENFORCED`。
`resource_limits_enforced`、`cleanup_verified`、`real_tokenizer_loaded`、
`benchmark_eligible`、`formal_evaluation_ready`、`net_benefit_established`、
`isolated`、`all_possible_second_rounds_validated` 均为 false。
