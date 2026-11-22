# 新校准：固定源码远端监督与本地串行主控实现结果

日期：2026-11-20，Asia/Shanghai。

状态：**本地固定源码执行四模块全部实现并通过 R3 最终独立审阅（APPROVE，0 P0/P1/P2，44 项新增组件回归全过，196 份基线文件保全）；真实 480 项远端执行尚待实时隔离预检（r2 范围与采集器已审阅通过）授权完成后启动。**

## 已完成的模块拼图

本阶段实现了 `2026-11-20-lora-recalibration-fixed-execution-design.md` 第 2 节余下的两个核心模块，与此前已验收的 `recalibration_execution_package.py`、`recalibration_execution_wire.py`、`recalibration_execution_evidence.py` 及原 `recalibration_validation.py` 形成完整本地闭环：

1. **`experiments/lora_necessity/recalibration_execution_remote.py`**（纯标准库远端可信监督层）：
   - 零仓库模块导入，可在远端通过 `<python3.12> -I -S -B -c` 独立运行，不将本地仓库或题库目录暴露给远端解释器搜索路径。
   - 在发出 `READY` 前建立唯一单调时钟锚点，校验 `LRB01` 启动帧、`config_sha256`、目标主机名/UID、解释器路径及 `validation_remote.py` 原始字节哈希，建立独占 `0700` `stage_root` / `batch_root` 与 `0600` 启动凭据，并在 10 秒上限内发出 `LRV01` 第一帧 `READY`。
   - 接收控制端 `grant` 与精确 `bundle_raw`（校验立即 EOF，拒绝多余字节或摘要不符；`grant` 只能收紧而绝不延长或重置 `READY` 前建立的 95 秒锚点）。
   - 以 `close_fds=True`、`pass_fds=()` 并在 `exec` 前关闭本地配置文件的方式启动独立会话 backend 子进程，保持 `seccomp` 过滤器文件描述符布局与预检一致；独立追踪直接子进程回收与进程组消失（共享 2 秒回收窗口），仅在结束后发出五帧 `LRV01` 流（总量严格不超过 8 MiB，超限绝不裁剪伪造 `DONE`）。

2. **`experiments/lora_necessity/recalibration_execution_controller.py`**（无环启动包与本地 15 批串行主控）：
   - 按 `preparation -> launch_config (config_sha256) -> 15 条精确 SSH 命令 -> launch_package (package_sha256)` 构造无哈希环完整包，现场核验 6 份数据文件、backend/worker 及 11 个本地 Python 模块的真实导入闭包与源码哈希，确保远端参数不含任何答案、角色或见证预测。
   - 先独占创建本地 `0700` `local_stage_root` 并落盘 `stage-claim.json`（永久消费单次 stage 身份），在 1800 秒总窗口（预留末尾 5 秒收尾，每批启动与授权前均要求剩余至少 102 秒）内串行驱动至多 15 批 SSH 握手与采集。
   - 正常参考/见证语义不匹配继续跑满 15 批并在报告中记录 `data_checks_satisfied=false`；任何程序错误、超时、回执缺失、输出超限或持久化失败立即停止后续批次，未知派发保守保留 `unknown_after_dispatch`。
   - `run_authorized`、`trusted_collection_accepted`、`dynamic_checks_passed` 及全部研究准入标志在合成测试与控制器原始输出中均保持 `false`。

## 本地验证证据

私有验证目录：`/private/tmp/lora-recalibration-controller-20261120-sSYngX`

| 验证项 | 结果 |
| --- | --- |
| `remote-red-19.log` / `.exit` | 19 项初始预期缺失实现失败，exit 1 |
| `controller-red.log` / `.exit` | 14 项初始预期缺失实现失败，exit 1 |
| `fresh-review.md` / `fresh-review-r2.md` | 首轮与次轮独立审阅原件保留（覆盖关闭管道 `EBADF`、时钟回退/清理期取消升级 `SIGKILL`、批次 `>=1` 中断时重建 `unknown_after_dispatch`、关闭描述符保留主取消信号） |
| `review-remote-r1-red-v2`、`review-controller-r1-red`、`review-remote-r2-red`、`review-controller-r2-red` | 全部审阅缺陷均先定向 RED 复现（exit 1），再最小修复通过 |
| `review-stack-r3-green.log` / `.exit` | 44 项合并回归（24 remote + 20 controller，含 15 批 480 项全流程及真实 `supervise_batch` 字节握手）全部通过，0 跳过，exit 0 |
| 基线文件保全 | `experiments/lora_necessity/` 既有 196 份文件逐字节未变 |
| 预检采集器 r2 `test_preflight_collector_r2` | 针对审阅意见加固后先 RED（7 失败）后 GREEN（7 通过），exit 0；`preflight-scope-review-r2.md` 审阅通过（APPROVE，0 P0/P1/P2） |
| `fresh-review-r3.md` | 最终独立代码与规格审阅通过（APPROVE，0 P0/P1/P2） |

已审阅通过的源码 SHA256：

- `recalibration_execution_remote.py`：`64f977d8ec1ab963866e9116078e6799d89a66d94fde0989cc445922990ad67c`
- `test_recalibration_execution_remote.py`：`dfbe068ca05f95e2533be238681c1b447367b35234770da64f50270963f4801d`
- `recalibration_execution_controller.py`：`b70b80e63cb029a2a722cfe1749f8c36aeb60f3164b5349378c60eaf480881b6`
- `test_recalibration_execution_controller.py`：`eacbecbb2d61da295281f582176622ecade3f8ab67cd06e3a2493afc238585d8`

## 当前研究结论与紧邻下一步

本地固定源码执行代码已齐备，但合成与人工边界测试不代表远端隔离实测通过，更不代表 LoRA 相比同信息文本有效。

紧邻下一步只需对已冻结的**单次实时隔离与运行时预检范围**（`20261120-recalibration-preflight-scope.md`，仅运行人工固定探针获取真实 10 字段 `expected_runtime`，不跑题库、不加载模型或 GPU）执行一次实测；拿到真实 `expected_runtime` 后即可直接组装并单次运行新版 480 项动态校验。
