# 新校准：已授权单次只读远端路径查询结果

日期：2026-11-20，Asia/Shanghai。

状态：**单次只读连接与身份/路径查询已完成；远端隔离预检、480 项动态验证和模型实验尚未执行。**

## 执行依据与边界

承接用户对上一轮阻断项（精确单次只读服务器查询）的明确“可以 继续”指令，
仅对已审阅冻结范围 `route-observation-scope-r2.json`
（SHA256 `7c8bf4e3a4200f6dcedfb00f03444304a957081f79867be5fa019a34c2decdea`）
通过 R4 审阅通过的 `route_observation_collector.collect_once`
（SHA256 `175515df2d32afbf7c6e13ef0c48e58223ce60d663e163b6dbdbe9c31af71a4b`）
执行唯一一次连接。

没有将此次授权扩大到隔离探针、题库执行、tokenizer 计数、模型加载、
GPU 查询或训练；没有修改 SSH 配置、known_hosts、凭据缓存或既有封存文件。

## 实测观察结果

| 观察项 | 实测结果 |
| --- | --- |
| 本地调用返回 | 正常返回，`status="OBSERVED"`，`observation_complete=true` |
| 持久化与候选回执绑定 | `persistence_completed=true`，`consumer_artifacts_verified=true` |
| 退出码与本地回收 | `returncode=0`，`child_reaped=true`，`local_group_absent=true` |
| 前后配置及运行库检查 | `precheck=true`，`postcheck=true` |
| 工作 / 回收耗时 | `0.981008417` 秒 / `0.000043959` 秒 |
| stdout / stderr 字节数 | `178` bytes / `0` bytes，无溢出 |
| 目标内核与预期身份 | `Linux`，目标主机名与预期数字 UID 一致（内部标识仅保存在私有回执） |
| 默认 `/usr/bin:/bin` 中命令 | `bwrap` 存在（`/usr/bin/bwrap`），`python3` 存在（`/usr/bin/python3`），`python3.12` 为 `null` |

默认精简 `PATH`（`/usr/bin:/bin`）中不存在 `python3.12` 符合既有环境布局：
目标机上的 Python 3.12 实验运行环境位于独立环境目录，不在系统 `/usr/bin` 下。
不能因此改用系统 `/usr/bin/python3` 充当实验解释器；下一阶段实时隔离预检必须
显式绑定既有 Python 3.12 环境根目录并实测其可执行文件哈希、bwrap/seccomp
过滤器及人工隔离探针。

## 私有证据与摘要

- 采集器独占记录目录：
  `/private/tmp/lora-route-collector-20261120-QF2RTX/route-attempt-7c8bf4e3a4200f6dcedfb00f03444304a957081f79867be5fa019a34c2decdea`
- 调用方独立捕获目录：
  `/private/tmp/lora-route-capture-20261120-kHScoI`

| 私有工件 | 字节数 | SHA256 |
| --- | ---: | --- |
| `attempt/claim.json` | 164 | `0786cc452c1594a309d26a557bc81a02c2ee747d2dee18a8b50098d93da49dd6` |
| `attempt/identity.json` | 418 | `665c521dd29f82abe66deecdbf559ec96da5992aa3fbda2ff7d6e48dcd3a51b2` |
| `attempt/stdout.bin` | 178 | `46e40812f8f18edc930b67433ffbc325dab28fc072e8decdedb8f5219f8181a2` |
| `attempt/stderr.bin` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `attempt/receipt.json` | 1573 | `354e03f36045af3907a7e09695a873e2dcb846144e04722c1df49d2dd7b967f3` |
| `capture/caller-invocation-claim.json` | 504 | `95d7ccfa47641b6fe159a8b533e608e4081c6aa8f43d7539ca54df821d6ce160` |
| `capture/caller-return.json` | 2400 | `8ff43fd43ea0d8ee703353281b7430d94be45078e5b1dc987d94d8c506d85ed2` |
| `capture/capture_once.py` | 9703 | `717ced8c4df28fabace58b6b3f4589a4f36b9de11e56b04d9fbfaa1ecb8d13be` |

采集器原封存的 48 份文件及 `route-observation-scope-r2.json` 均逐字节未变。

## 结论边界与后续步骤

本次观察只确认当前两跳严格认证路线连通、目标身份一致以及 `/usr/bin/bwrap`
存在；`remote_isolation_qualified`、`trusted_collection_accepted`、
`benchmark_eligible`、`formal_evaluation_ready`、`net_benefit_established`
和 `research_goal_complete` 全部保持 `false`。

后续推进路径：

1. 本地完成固定源码执行链剩余两个模块：`recalibration_execution_remote.py`
   与 `recalibration_execution_controller.py`，通过 TDD 与独立代码审阅。
2. 冻结并独立审阅显式 Python 3.12 运行环境 + `bwrap`/`seccomp` 人工探针的
   一次性实时预检范围，在获得授权后实测 `expected_runtime`（含精确 FD 布局）。
3. 绑定实测 `expected_runtime` 单次执行新版 480 项固定源码验证与 120 行
   真实 tokenizer 长度测量，随后推进配对诊断与同信息 LoRA 对照。
