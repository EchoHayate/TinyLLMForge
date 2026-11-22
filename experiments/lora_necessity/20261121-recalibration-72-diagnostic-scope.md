# 新校准 Gate G3：72 次 Qwen3-8B 配对校准诊断冻结启动包与单次范围

日期：2026-11-21，Asia/Shanghai。

状态：**本地模块（排程/评分、远端两阶段执行器、受信主控）与 R2 冻结启动包均已完成并通过独立代码与包审阅（APPROVE，0 P0 / 0 P1 / 0 P2）；尚未连接远端启动 72 次 GPU 生成，等待明确单次执行授权。**

## 1. 前置关卡状态核对

| 前置关卡 | 状态 | 关键证据与哈希 |
| --- | --- | --- |
| 实时 `bwrap` + `seccomp` 隔离预检（r4） | **已通过** | `preflight-canonical.json`：`89bbaaec675798df3ec0d7f15c6d877882984f4730d716f5cef0f6791c0b9aef` |
| Gate G1：15 批 480 项固定源码动态验证 | **已通过并独立复核** | `192/192` 参考与 `36/36` 见证全过；`480-stage-acceptance.json`：`9a346f58cf298a3209e146f1e3b9fc730753791b1e78c81df3f116d2b3a07b54` |
| Gate G2：120 行真实冷进程 Qwen3-8B 分词核验 | **已通过并独立复核** | `120/120` 零超限（最大输入 `2734 <= 4096`，最大目标 `145 <= 256`）；`expected_tokenizer_sha256`：`7743a3dacfe43382819ca9e9d87993c92ad45f31d9b8afa432b425437d6f2136`；`120-tokenizer-acceptance.json`：`5f109af2b39d3539e08320fd82741d1d32ca277918b040febb994b527bb547b6` |

## 2. Gate G3 实现与两阶段物理防泄漏设计

1. **核心排程与评分模块 `recalibration_diagnostic_scoring.py`**（SHA256 `c45bb0b067ac3d198aa1f84610f63b5d10d1788fd353f49d88b6dfe19f39615b`）：
   - 依据 `prompts.json` 与 `supervision_v1.json` 构造确定性 24 episode / 72 次贪婪生成对称排程（`C01..C12` × `base`/`text`，记忆顺序与 `feedback`/`retry` 分支顺序完全配平）。
   - 执行单函数纯 AST 语法/签名与字节上限校验（`<= 4096` UTF-8 字节且 JSON 编码 `<= 3584` 字节），构造共享首轮答案的二轮 `visible_tests` / `invalid_source` / `results_withheld` 消息，按五状态（`PASS`、`SEMANTIC_FAIL`、`EXECUTION_FAIL`、`UNKNOWN`、`INVALID_EVIDENCE`）分类并联合核验 Phase 1 与 Phase 2 公开状态一致性，输出四级校准结论（`INVALID_PROTOCOL`、`INCONCLUSIVE_INCOMPLETE`、`INCONCLUSIVE_CALIBRATION`、`DEVELOPMENT_DIAGNOSTIC_USABLE`）。
2. **远端 Phase 1 执行器 `recalibration_diagnostic_remote.py`**（SHA256 `eaddbce9f97c1a89e4fa572911fb4f89398a2f4a4064a90f25dcb0462cc3f09d`）：
   - 自动选取空闲 1× A100 80GB GPU（`free >= 65536 MiB`、`used <= 512 MiB` 且无在跑计算进程），逐文件核验 Qwen3-8B 全部 12 个配置/分词/权重分片 SHA256，以 BF16、`attn_implementation="eager"`、`use_cache=False`、贪婪解码（`input <= 4096`、`output <= 256`）执行 72 次观察式生成，并在 `bwrap` + `seccomp` 子进程中执行公开用例 `P01`/`P02`（若 `P01` 因异常提前停止，自动补跑单条 `P02`），生成前后核验模型权重与 CPU/CUDA RNG 零漂移。
   - **零私有用例传输**：Phase 1 载荷（`235,489` bytes，SHA256 `ceb01ea71ed6a0a908111d15742955aee346697f0cee3791e473bedf278491dc`）仅含 13 个固定模块、`prompts.json`、`supervision_v1.json`、`validation_remote.py` 与 `validation_task_worker.py`，完全不含 `oracles.json`、私有用例 `H01..H06`、参考解或预置缺陷解。
3. **本地受信主控 `recalibration_diagnostic_controller.py`**（SHA256 `96d7c9890a8cf9aa094c210891a32f05afb6964cb4a3ac3210fb28cd499ac1e9`）：
   - 先运行 Phase 1 并将全部 72 份生成候选落盘封存（`phase1-result.json` + `fsync`），再启动 Phase 2（至多 18 批 32 项四字段 `LORA_REMOTE_VALIDATION_BUNDLE` 送入远端 `bwrap` + `seccomp` 隔离执行；对正常已回收的候选异常停止支持确定性续批，对未回收或基础设施异常立即中止后续批次），最后在本地对照 `oracles.json` 完成评分。
   - 合并单元测试套件（`test_recalibration_diagnostic_scoring` + `test_recalibration_diagnostic_controller`）`11/11` 全绿通过。

## 3. R2 冻结启动包摘要（私有敏感字段已脱敏）

- 私有包目录（`0700`）：`/private/tmp/lora-recal-72-diag-package-r2-20261121-os4r12fi`
- 未消费运行根目录父级（`0700`，当前为空 `[]`）：`/private/tmp/lora-recal-72-diag-runs-r2-20261121-c6nep1xn`

| 工件 / 绑定字段 | 数值 / SHA256 |
| --- | --- |
| `run_id` | `f0b49750e43921321fccb259cac4f5ca` |
| `package_sha256` | `59466a01ca17da36b48aa6bda13867c8feb0f2323c9eb33e76b7da58da53e4d5` |
| `phase_one_config_sha256` | `79c5ac8069125626b20472f4ee3f5fb83f9447d0610ea031641bb2f2e742b93f` |
| `schedule_sha256` | `af0acce971a6cebe63a2f57e32ef219096009b9e7da21e5a8788fb314de55251` |
| `one_time_scope_sha256` | `a2d335001bc5b37a8ae7f61af2ac29988e597e6fec48036095310d1b43387cdd` |
| `diagnostic-launch-package-full.json` | `d1dd91d174fe26994ba3e913e184ff856a4f3ce2870718ea4b7ea8af0ee183e8` |
| `diagnostic-dispatch-metadata-r2.json` | `34ba6dd1c72dcc5096cc413155380115075167537391845f9a372d9bdb5cb17b` |
| `diagnostic-package-redacted.json` | `9b44beaf939d451bc61e8493479ce9772903ec01ddfaa38cf178f8883dd31da4` |
| `execute_72_diagnostic_once_r2.py` | `c5a687ac5f5a8363e9c24e94d4219ea07de91095593541f3f32e99a483dceff4` |
| `independent-72-diagnostic-package-review-r2.md` | `733bde019583cbcfa2df706eb681e249571408f0158bdd90130c61697753e5eb`（`APPROVE`，0 P0/P1/P2） |

## 4. 权限边界与待授权单次操作

所有治理标志（`trusted_collection_accepted`、`run_authorized`、`benchmark_eligible`、`formal_evaluation_ready`、`net_benefit_established`）均保持 `false`。本次单次执行请求仅覆盖运行 `execute_72_diagnostic_once_r2.py` 一次（1 张空闲 A100 80GB 上完成 72 次 Qwen3-8B 贪婪生成 + 公开用例反馈，随后完成 `<= 18` 批 8 用例隔离评分；总墙钟上限 `5400` 秒，失败不自动重试），不包含任何 LoRA 训练或 32 题正式评估。
