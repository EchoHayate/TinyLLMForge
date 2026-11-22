# 新校准：15 批 480 项固定源码真实动态验证通过结果

日期：2026-11-21，Asia/Shanghai。

状态：**15 批 × 32 项 = 480 项真实隔离动态验证已单次全部完成并通过独立离线回执重建审阅（APPROVE，0 P0/P1/P2）；192/192 参考位置与 36/36 具名缺陷见证全部匹配预期。**

## 1. 实测核心结果

按已审阅冻结的启动包（`stage_id=8b2bf6a17d05c52691aafc39de874718`，`package_sha256=a92441a45b6fc2e227684490597bdf62f720997253bac6d0ce0b39438f2eaf8d`）执行唯一一次 15 批串行运行，总墙钟耗时 `57.380` 秒：

| 指标 | 计划分母 | 实测结果 |
| --- | ---: | ---: |
| 串行批次完成数 | 15 | **15 / 15**（全部 `done.status="completed"`，SSH `returncode=0`） |
| 单项隔离执行完成数 | 480 | **480 / 480**（`unknown_after_dispatch=0`，`not_dispatched=0`） |
| 参考实现一致性（`reference_a` + `reference_b`） | 192 | **192 / 192 匹配正确期望**（C01–C12 每题 16/16） |
| 具名缺陷见证（`initial` + `M01` + `M02`） | 36 | **36 / 36 匹配缺陷预测且异于正确期望**（C01–C12 每题 3/3） |
| `matrix_complete` / `data_checks_satisfied` | `true` | **`true` / `true`** |

## 2. 独立重建与 macOS 僵尸进程组探测说明

- 独立复核（`/private/tmp/lora-recal-480-capture-20261121-3a3R3V/independent-480-stage-review.md`，SHA256 `1f30b103a0c8ea277897a06cc4ec6228fcf3f6bb1a01d5809b5e34bfe1f7ff5d`）仅凭本地冻结的 6 份数据文件、`validation_remote.py`、`validation_task_worker.py` 及保存的 15 份原始 `wire.bin` 离线调用 `build_evidence(...)`，逐字节复现了 `execution-evidence.json`（SHA256 `454c004f70b3d476f86aaa114fc6848d7f610e147572883134daa7a6606173e5`，规范摘要 `340ec456c90365b09526b2ff1f32706511780afdf298e26aa5b116016131b023`）。
- 原始 `controller-summary.json` 中记录了一条 `secondary_errors: [{"phase": "ssh_cleanup_probe", "type": "PermissionError"}]`，原因已核实：在 macOS XNU 内核上，当会话首进程已退出但尚未被 `waitpid` 回收（处于 `SZOMB` 僵尸态）时，`os.killpg(pid, 0)` 返回 `EPERM`（`PermissionError`）；紧接着 `reap_once` 完成 `wait` 回收（`returncode=0`）并在下一轮 `killpg(pid, 0)` 得到 `ESRCH`（`group_absent=true`），因此 15 批 `ssh_status` 全部为 `"completed"`。原始 `controller-summary.json` 原样保留不改写，主控独立接受记录另存于私有 `480-stage-acceptance.json`（SHA256 `9a346f58cf298a3209e146f1e3b9fc730753791b1e78c81df3f116d2b3a07b54`，`trusted_collection_accepted=true`，`dynamic_checks_passed=true`）。

## 3. 结论边界

本阶段证明新版 12 道校准题（`recalibration_v2_r1`）的 24 份参考实现、36 份缺陷代码与 96 个测试用例在真实 `bwrap`+`seccomp` 隔离环境中行为完全符合静态设计预期。本次未加载 tokenizer 或预训练模型，不代表模型长度预算或 LoRA 收益结论。
