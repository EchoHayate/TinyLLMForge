# 新校准 Gate G3：R2 单次启动复盘与 R3 冻结启动包

日期：2026-11-21，Asia/Shanghai。

状态：**R2 单次执行已在 `MODEL_LOAD` 阶段（零生成、零 Phase 2 批次）干净停机并完整保留证据；定位并修复 PyTorch `CUDACachingAllocator` 惰性初始化顺序问题后，R3 模块（12/12 测试全绿）与 R3 冻结启动包已通过独立代码/包审阅（`APPROVE`，0 P0 / 0 P1 / 0 P2），等待单次启动授权。**

## 1. R2 单次执行实测与根因复盘（未自动重试）

按用户单次授权执行 `execute_72_diagnostic_once_r2.py`（`run_id=f0b49750e43921321fccb259cac4f5ca`），总墙钟 `47.708` 秒，`binding_complete=true`，`secondary_errors=[]`，`post_ssh_closure_ok=true`：

- **已通过的远端前置检查**：
  1. 两跳隔离 Python 引导与 `READY` 握手正常；
  2. 空闲 `NVIDIA A100 80GB PCIe` 自动选取成功（`index=2`，`free_mib=80979`，`used_mib=71`）；
  3. Qwen3-8B 全部 12 个配置/分词/safetensors 分片 SHA256 逐字节校验通过（`checkpoint_verify_ns = 36,983,986,254`，约 `36.98` 秒）；
  4. `AutoTokenizer` 加载并通过 `expected_tokenizer_sha256`（`7743a3dacfe43382819ca9e9d87993c92ad45f31d9b8afa432b425437d6f2136`）核验。
- **停机位置与根因**：
  - `phase1-result.json` 回报 `bootstrap_failure = {"phase": "MODEL_LOAD", "error_type": "RuntimeError", "message": "Invalid device argument "}`，`phase_one_records = []`（零次模型生成、零次 Phase 2 调用）。
  - 根因：`torch.cuda.is_available()`、`torch.cuda.device_count()` 与纯 CPU `AutoTokenizer.from_pretrained(...)` 均不会触发 PyTorch `torch.cuda._lazy_init()` 初始化 `CUDACachingAllocator`，因此在尚未调用 `torch.cuda.init()` / `torch.cuda.get_device_properties(device)` / `torch.cuda.synchronize(device)` 前直接调用 `torch.cuda.reset_peak_memory_stats(device)` 会因分配器设备表为空而抛出 `RuntimeError: Invalid device argument `。
- **R2 原件完整保留**：
  - `/private/tmp/lora-recal-72-diag-runs-r2-20261121-c6nep1xn/f0b49750e43921321fccb259cac4f5ca`（`phase1-wire.bin` SHA256 `42712b6e9f74e81214ba0ac4e9954de9f111341f8b5b06ab327c0d12d2f9cbe5`）
  - `/private/tmp/lora-recal-72-diag-capture-20261121-upvstfvj/diagnostic-caller-return.json`

## 2. R3 修复与独立核验结果

1. **`recalibration_diagnostic_remote.py`**（SHA256 `643c87d243d6212707f8857b35640ea10e0310fd7dd018d6929e33b12f6786aa`）：
   - 对齐既往已实测通过的 `development_remote_driver.py` 与 `generation_core.py`，抽离 `_initialize_cuda_and_load_model`，显式按序执行 `torch.cuda.init()`、`torch.cuda.get_device_properties(device).major == 8`、`torch.manual_seed(42)`、`torch.cuda.manual_seed_all(42)` 与 `torch.cuda.synchronize(device)` 后再调用 `torch.cuda.reset_peak_memory_stats(device)` 并加载 BF16 Qwen3-8B。
   - 在 `bootstrap_failure` 中补充最内层帧 `lineno` 与 `function` 字段。
2. **`recalibration_diagnostic_controller.py`**（SHA256 `af1e8a951cd3728513c28896dc673b9594cb784c4312a4748c5bc64ff052bcee`）与测试套件：
   - 新增状态化未初始化 CUDA 拦截单测 `test_remote_cuda_init_runs_before_reset_peak_memory_stats`（先复现 RED 再验证 GREEN），合并测试套件 `12/12` 全绿通过。
3. **R3 冻结启动包与独立审阅（`APPROVE`，0 P0 / 0 P1 / 0 P2）**：
   - 私有包目录（`0700`）：`/private/tmp/lora-recal-72-diag-package-r3-20261121-u2q30gad`
   - 未消费运行根目录父级（`0700`，当前为空 `[]`）：`/private/tmp/lora-recal-72-diag-runs-r3-20261121-4tyx4l88`

| 工件 / 绑定字段 | 数值 / SHA256 |
| --- | --- |
| `run_id` | `b0dc5a77fa55f0b6a2a169f68a529673` |
| `package_sha256` | `bd66c6bbd90e0237cef75e693b12dec068b1064777011a8350865ac3e5170972` |
| `phase_one_config_sha256` | `ed959d1ecfb06a25482e3d9930fb58db540642e15d112ca9933bb70ace505732` |
| `schedule_sha256` | `3f682693a7f0eb683f34149813648a7026926895faa8f931da891f30a95fd98b` |
| `one_time_scope_sha256` | `32e57ef8d2e420fb92d250363db48a780afaf50e97c889e2b3d65d0965b5bf9a` |
| `diagnostic-launch-package-full.json` | `619dfb19056a96fda63272e1fa69099889a6c554791682a9fc3b5003a4698fb8` |
| `diagnostic-dispatch-metadata-r3.json` | `f5eb8639c98f7491ac86d7f3d6f60b4bd2e811ac42c421f32f136a3116e7aa7b` |
| `diagnostic-package-redacted.json` | `35ee08a9066023fd67ae4490efca4a657f122e4ea6f887d785e65ec9810e31dc` |
| `execute_72_diagnostic_once_r3.py` | `7275ad025770517baa3a84507054ceda37ff80183ac9a3bfe5b7cd76e000ae9a` |
| `independent-72-diagnostic-package-review-r3.md` | `c00aaa25d89eb78e5952e60a05b8a583f1860f39e8aab2eceff8b38e94edd86f`（`APPROVE`，0 P0/P1/P2） |
