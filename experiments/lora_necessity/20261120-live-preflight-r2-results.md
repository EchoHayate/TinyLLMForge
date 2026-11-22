# 新校准：实时预检 r2 实测结果（venv 符号链接路径保护误拒）与 r3 修正

日期：2026-11-20，Asia/Shanghai。

状态：**r2 单次预检已实际执行并完整保留失败回执（未进入 `bwrap` 探针阶段）；根因已查明并完成 r3 探针与采集器修正（待单次重跑授权）。**

## r2 实际发生与根因定位

在获得用户明确授权后，使用 `preflight_collector_r2` 对 `live-preflight-scope-r2.json`
（SHA256 `48abf135477349a5292d154a69b4ecd89f0c08117ce9b14f3d0cee55399063e9`）发起唯一一次连接：

- 两跳严格认证连接与目标 Python 3.12 启动成功，耗时 `1.804102208` 秒，`returncode=1`，本地子进程与进程组干净回收（`child_reaped=true`，`local_group_absent=true`），前后配置校验均通过。
- `stdout` 为 `0` 字节，`stderr` 为 `173` 字节（SHA256 `64e0b74a0d8611034f907ad6975b4779b8a4bd5b7e270aa69e549ece91b04526`），精确报错位于 `preflight_probe_remote_r2.py` 第 160 行：`RuntimeError: unexpected remote python interpreter`。
- **确切根因**：目标机候选路径是 Python 虚拟环境（`venv`），其 `bin/python3.12` 是指向底层基础 CPython 3.12.13 安装目录（`sys.base_prefix`）的符号链接。而 `validation_remote.py` 第 274 行本来就通过 `Path(sys.base_prefix).resolve()` 解析基础运行时根目录并只读挂载到 `/runtime`。`preflight_probe_remote_r2.py` 误要求 `os.path.realpath(sys.executable)` 仍留在上层 `venv` 目录内，导致符号链接解析后在启动 `bwrap` 前即主动退出。
- 按单次消费规则保留原始 r2 失败目录与回执（`caller-return.json` SHA256 `5b1ecf71d89bbbd7222126c93861c86217c55f1c8871dc87dc4f098c73c2a8e4`），未自动重连或覆盖。

## r3 定向修正与本地验证

新增 r3 探针、范围与采集器（位于 `/private/tmp/lora-preflight-capture-20261120-ShtfdI/`，原 r2 封存文件逐字节未变）：

1. `preflight_probe_remote_r3.py`（SHA256 `df34105a0859ce9dece8389c77c0c7800df957eb78e72a14f58c1368de90cb7f`）：核对 `platform.python_version() == "3.12.13"`，并与 `validation_remote.py` 第 274 行完全一致地按 `Path(sys.base_prefix).resolve()` 解析真实基础解释器根目录与可执行文件（同时直接避免把含训练包的 `venv` 目录挂入隔离区）。
2. `live-preflight-scope-r3.json`（SHA256 `7e7b3d0cdad51228d6b2279324504bb6f13add21dd56202230a4c382bb0804e3`）：保持完全相同的两跳严格安全参数、限制与禁令，状态为 `DRAFT_NOT_AUTHORIZED_OR_EXECUTED`。
3. `preflight_collector_r3.py`（SHA256 `0255213814679055dd0f86cf43c957d2eafab8ffc36d8648b7a74cc162f25b9e`）及 `test_preflight_collector_r3.py`：新增真实本地 `venv` 符号链接到独立 `base_prefix` 目录的复现测试，先 RED（5 失败）后 GREEN（5 通过，exit 0）。
