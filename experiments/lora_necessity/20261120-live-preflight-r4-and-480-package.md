# 新校准：实时隔离预检 r4 通过与 15 批 480 项完整启动包就绪

日期：2026-11-20，Asia/Shanghai。

状态：**单次实时隔离与运行时预检（r4）已全部通过；基于实测 `expected_runtime` 的 15 批 480 项完整启动包已在本地构造并校验通过（`DRAFT_NOT_AUTHORIZED_OR_EXECUTED`，尚未派发执行）。**

## 1. 实时隔离与运行时预检（r4）实测结果

在获得用户明确授权后，使用 `preflight_collector_r4` 对 `live-preflight-scope-r4.json`
（SHA256 `ca437b1332afa1a03dd4e03a28a9293e4b1e5391d544087c5867cb17ad99826c`）发起唯一一次连接：

| 指标 / 检查项 | 实测结果 |
| --- | --- |
| 采集状态 | `status="PREFLIGHT_OBSERVED"`，`preflight_complete=true`，`persistence_completed=true` |
| 进程与配置校验 | `returncode=0`，`child_reaped=true`，`local_group_absent=true`，`precheck=true`，`postcheck=true` |
| 耗时与流大小 | `4.094054458` 秒；`stdout=1267` bytes，`stderr=0` bytes |
| 实测 Python 版本与哈希 | `3.12.13`，`python_sha256=7d43f6e86a6c6dd12005ec77eb2055f1be3f1bb3adedf8afe0a87973fa7371ce` |
| 实测 `bwrap` 哈希 | `6a43389e0b1a3042b5bf26cc9f55d27cedb536ae5d5b3b2b3879e65d6fad3166` |
| 实测 `seccomp` 过滤器哈希与 FD | `7e9670058f0b7674edadea76cb7c048333c80bad97e3fed77f8d1790faba67e2`，`--seccomp 3` |
| 人工固定探针 13 项检查 | 4 类命名空间隔离、`NoNewPrivs=1`、`Seccomp=2`、4 处 `--ro-bind` 只读挂载、根挂载表干净、宿主目录与外层合成假文件不可见、`socket`/`fork`/`clone3` 返回 `EPERM`、2 秒超时回收、64 KiB 输出上限截断回收全部通过 |

私有捕获目录：`/private/tmp/lora-preflight-r4-capture-20261120-Cd46zU`
- `caller-return.json` SHA256：`35d2b5c0968c61c866256fa19802e611840ec9cfd4a160d5049691bb12ed3a7c`
- `live-preflight-receipt.json` SHA256：`941c3423b5e09d85b8c9080d2041a9873f8b273308bae2dcf7e6406624ea6006`

## 2. 15 批 480 项固定源码验证完整启动包

基于上述实测 `expected_runtime`（规范 JSON SHA256 `89bbaaec675798df3ec0d7f15c6d877882984f4730d716f5cef0f6791c0b9aef`），已在本地通过 R3 审阅批准的 `recalibration_execution_controller.build_launch_package` 与 `validate_launch_package` 完成无哈希环完整启动包构造与双向重建校验（未启动任何连接，本地 `local_stage_root` 尚未消费）：

- 私有包目录：`/private/tmp/lora-recal-480-package-20261120-ty5Ynv`
- `stage_id`：`8b2bf6a17d05c52691aafc39de874718`
- `preparation_sha256`：`76872dba23498f4f613dc9ecdb8a1655b2d0c6c7c117a9bcba5173cb7ad14b21`
- `config_sha256`：`64bbea023e4199ac89c71160f2cd5f62a8b21dfa315759c97f9015c2476ff90e`
- `package_sha256`：`a92441a45b6fc2e227684490597bdf62f720997253bac6d0ce0b39438f2eaf8d`
- `one_time_scope_sha256`（`stage-execution-scope-r1.json`）：`2a4b1ae460d67cc593de2f19345f1958cc03dcd5a86b5a69eda576eff37ace52`
- 单次驱动入口（经独立审阅加固 r2）：`execute_480_stage_once_r2.py`（SHA256 `a2f00839f5f8eef9ef1d4828bdb12defc4a0dfa3b1745475643ffac7fb504fab`）
- 脱敏结构摘要（r2）：`launch-package-summary-redacted-r2.json`（SHA256 `2618e713c830460822aba9bd068602aeb7fb3e3d955d10c6c2cb28532daa7b23`）

### 执行范围与边界说明

- **规模与内容**：固定 12 道新校准开发题 × 5 份固定源码（2 参考 + 3 错误版本）× 8 个用例 = **480 项（15 批 × 32 项）**，检验 192 个参考位置与 36 个具名错误见证；远端只接收四字段请求（`request_id`/`source`/`entry_point`/`args`），不含任何答案、角色或见证预测。
- **隔离与时限**：每项在 `bwrap --unshare-all --seccomp 3` 内运行（单项 2 秒、每批 backend 90 秒、监督 95+2 秒、SSH 110+2 秒、整段 1800 秒总上限），任何基础设施/超时/回执异常立即停止后续批次且不自动重跑。
- **不做事项**：不加载 tokenizer 或预训练模型，不查询或占用 GPU，不进行模型生成或 LoRA 训练。
