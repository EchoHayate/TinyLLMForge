# 新校准 Gate G3：72 次 Qwen3-8B 生成（Phase 1）完成结果与 Phase 2 CPU 最终评分包

日期：2026-11-22，Asia/Shanghai。

状态：**Phase 1（12 题 × `base`/`text` × `initial`/`feedback`/`retry` 共 72 次 Qwen3-8B GPU 生成与公开用例反馈）已 100% 完成并独立核验通过（`72/72` 候选全部语法签名合法，权重/分词/RNG 前后零漂移）；Phase 2 第 0 批在零候选执行前因 `sshd` 继承文件描述符触发 `verify_runtime` 拦截停机，现已修复为 `close_fds=True` 子进程启动并通过独立审阅（`APPROVE`，0 P0/P1/P2），等待单次授权运行这 ~60 秒的 18 批纯 CPU 最终隔离评分（无需重跑 GPU）。**

## 1. Phase 1（72 次 Qwen3-8B GPU 贪婪生成 + 公开用例反馈）实测结果

在 `run_id=b0dc5a77fa55f0b6a2a169f68a529673` 中，Phase 1 以 `returncode=0`、`bootstrap_failure=null` 一次跑满全部 24 个 episode（72 次贪婪生成，零私有用例传输），并已封存于本地 `phase1-wire.bin`（SHA256 `353172ce9a39649b2a25588dd28db48744283edd152dc8923cd35d2b1010a3ba`）：

| 实测指标 | `base`（无文本示例） | `text`（同信息可信文本示例） |
| --- | ---: | ---: |
| 计划 / 完成生成次数 | **36 / 36** | **36 / 36** |
| 单函数 AST 语法与签名有效数 | **36 / 36（100%）** | **36 / 36（100%）** |
| 输入 token 总数 | `20,687` | `91,230`（**`4.41×`**） |
| 输出 token 总数 / forward 总次数 | `3,449` / `3,449` | `3,358` / `3,358` |
| GPU 生成墙钟耗时 | `378.229` 秒（6 分 18 秒） | `2,031.403` 秒（33 分 51 秒，**`5.37×`**） |
| 生成峰值 `allocated` / `reserved` 显存 | `15.59 GiB` / `67.20 GiB` | `17.91 GiB` / `78.45 GiB` |
| Phase 1 公开用例（`P01`/`P02`）隔离执行耗时 | `2.755` 秒 | `2.902` 秒 |
| 首轮 `initial` 两公开用例全过 / 公开揭示错误任务数 | **`6` 题全过 / `6` 题揭示错误** | **`4` 题全过 / `8` 题揭示错误** |

完整性核验：
- 基座模型初始与结束张量摘要一致：`f544e8ae7e4fb85aa189a013cd435bdd7872bfbfd18e10974d37d41a811e4e60`
- 分词器状态摘要一致：`7743a3dacfe43382819ca9e9d87993c92ad45f31d9b8afa432b425437d6f2136`
- CPU / CUDA 随机数状态零漂移：`rng_unchanged = true`

## 2. Phase 2 停机复盘与 `close_fds=True` 修复

- **停机原因**：Phase 1 落盘封存后进入 Phase 2 第 0 批时，顶层 `sshd` 派生的 Python 进程直接在当前进程内 `exec` 了 `validation_remote.py`，未像 `recalibration_execution_remote.py` 及 Phase 1 `_invoke_backend_bundle` 那样通过 `subprocess.Popen(..., close_fds=True)` 关闭 `sshd` 继承的高号文件描述符，导致 `os.memfd_create` 未拿到 FD `3`（`--seccomp 3`），在执行任何候选代码前被 `validation_remote.py:241`（`verify_runtime`）干净拦截（`stdout=0` 字节，零候选执行），主控随即中止后续批次并保留全部 R3 原件。
- **修复与独立核验**：
  1. `recalibration_diagnostic_controller.py`（SHA256 `60f091f7a1359585f870c72e1660370fc3108f5c7d679658250c4231e9d9daf9`）将 `_build_phase_two_ssh_argv` 对齐 `recalibration_execution_remote.py` 与 Phase 1 `_invoke_backend_bundle`，通过 `subprocess.Popen(..., close_fds=True, pass_fds=(), start_new_session=True)` 启动 `validation_remote.py`；
  2. 新增 `build_phase_two_scoring_package` 与 `run_phase_two_scoring_stage`，先离线校验 `phase1-wire.bin` 五帧密码学绑定，再在独立新目录对已封存的 72 份候选（`576` 个有效位置，`18` 批 × `32` 项）仅运行纯 CPU `bwrap`+`seccomp` 最终评分（无需重跑 40 分钟 GPU 生成），`13/13` 单元测试全绿通过。

## 3. Phase 2 纯 CPU 最终评分冻结包摘要（`APPROVE`，0 P0 / 0 P1 / 0 P2）

- 私有包目录（`0700`）：`/private/tmp/lora-recal-72-diag-p2-package-20261122-8pbuhdd0`
- 未消费运行根目录父级（`0700`，当前为空 `[]`）：`/private/tmp/lora-recal-72-diag-p2-runs-20261122-ss6hita9`

| 工件 / 绑定字段 | 数值 / SHA256 |
| --- | --- |
| `stage_id` | `dbf831b5f7a6e0f82fd36831d34a87d1` |
| `source_run_id`（绑定已封存 Phase 1） | `b0dc5a77fa55f0b6a2a169f68a529673` |
| `sealed_phase_one_wire_sha256` | `353172ce9a39649b2a25588dd28db48744283edd152dc8923cd35d2b1010a3ba` |
| `package_sha256` | `887c25696302c439b2e42a3e029cd9def240d0542e171eaa67593cbbaf1cca17` |
| `one_time_scope_sha256` | `0bdb7fc027fbac073eedf79adfcf026edfd8155bc06b70588ce581e3a2b93512` |
| `diagnostic-phase2-package-full.json` | `dcc55e5d04be74722f43dc5016edf105b40b4f3f37eb53dfc9e539613844b1bf` |
| `diagnostic-phase2-dispatch-metadata-r1.json` | `d915508510dccf41df8c27e12460222bfa7a13610230c77ed022b25fb3915396` |
| `diagnostic-phase2-package-redacted.json` | `fda5e6b72b6b1581e79e1d329e4e06af1ede7ddb90049729393256663dc4c69a` |
| `execute_72_diagnostic_phase2_once_r1.py` | `8888656f2eb4b61f63e1b6e803c2e75105ad20c6e54c071d3cbe74e8696541d5` |
| `independent-72-diagnostic-phase2-review-r1.md` | `b4097cbb46b4a7d53d2bf3e3b320e6a99983aac4ad6ac51fa141abdb25685f39`（`APPROVE`，0 P0/P1/P2） |
