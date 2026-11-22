# 新校准：120 行真实 Qwen3-8B Tokenizer 长度审计包与范围

日期：2026-11-21，Asia/Shanghai。

状态：**已由通过验收的 480 项真实动态证据生成 120 行清单与完整启动包，并通过 R2 独立审阅（APPROVE，0 P0/P1/P2，状态保持 `DRAFT_NOT_AUTHORIZED_OR_EXECUTED`）；待单次执行授权。**

## 1. 清单来源与构造分母

基于已通过独立复核的 480 项动态验证原始证据（`480-stage-acceptance.json` SHA256 `9a346f58cf298a3209e146f1e3b9fc730753791b1e78c81df3f116d2b3a07b54`），在本地重建 `validation_report`（SHA256 `34f8a61e662fef871b5fbf4b9bb959c53995540a105f31841ea1be5513d39ac9`）并提取每题真实公开用例反馈，生成固定 **120 行**计数清单（`inventory_sha256=d60b90bd8589e7e46793449d1908cb1b1f7ca24c331411b4bc87e5199f4083e2`）：

- `first`（12 题 × `base`/`text` 首轮提示）：**24 行**
- `references`（12 题 × `base`/`text` × `reference_a`/`reference_b` 答案含唯一 EOS）：**48 行**
- `sampled_second`（12 题 × `base`/`text` × `feedback`/`retry` 二轮提示）：**48 行**

保持预算上限不变：输入 `<= 4096` tokens，答案（含唯一 EOS）`<= 256` tokens，`enable_thinking=False`。

## 2. 冻结包身份与隔离边界

私有包目录：`/private/tmp/lora-recal-120-tok-package-20261121-FpIo4a`

- `run_id`：`81b42b40fc9ffdfb506e0198bf7d8b96`
- `package_sha256`：`3a08217bad0876993f119ccfc26dcc545fa023d372d6107ccff14db04590346c`
- `config_sha256`：`bb4d74eb90c4935d13c6b33cec3bd62b04d2edf8bd2d7101c484064c0dec9f95`
- `projection_sha256`：`e1d8bfdc06ae348282cf580fc94f8d0e7614287c1aead01e9cd6e3dac38d4729`
- `profile_sha256`：`70a8843848b5cf07926bd46ff2b78345841792194e735bd04496b51fbcb172cb`
- `one_time_scope_sha256`：`c1cfb5fb10e2ef198c499b003e80642599c2fc5006fa9737b31bb70ad9b59b8f`
- 单次驱动入口（R2 审阅通过）：`execute_120_tokenizer_once_r2.py`（SHA256 `3bc294be469e8461dcc440902487dcbcbed71ca3df0b378b89cfe5f18fb8c4a9`）

### 运行边界

- **最小计数投影**：远端仅接收 120 行 `{ordinal, messages, answer, row_sha256}`、7 个纯 tokenizer 会话模块及 `profile.json`，不上传完整 oracle 或动态私有回执。
- **冷进程与禁用模型/GPU**：远端以全新 `-I -S -B` Python 3.12.13 冷进程启动，安装 `seccomp` 拒绝 `fork/vfork/clone/clone3` 与多线程，`RLIMIT_CORE=(0,0)`，设置 `USE_TORCH=0`、`CUDA_VISIBLE_DEVICES=""`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`；仅从新建 `0500`/`0400` 只读五文件快照加载 `PreTrainedTokenizerFast`，**不加载任何模型权重、不使用 GPU、不执行任何题目代码、不训练**（工作硬上限 1200 秒，预计约 4 分钟完成）。
