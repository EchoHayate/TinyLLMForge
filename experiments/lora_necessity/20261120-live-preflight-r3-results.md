# 新校准：实时预检 r3 实测结果（`bwrap` 临时根挂载断言修正）与 r4 就绪

日期：2026-11-20，Asia/Shanghai。

状态：**r3 单次预检已实际跑完三组 `bwrap`+`seccomp` 子进程探针（耗时 `4.337s`，三次 `expected_runtime` 完全一致），仅被探针自身对 `/` 的写测试误拒；r4 已对齐 `validation_remote.py` 的四个 `--ro-bind` 挂载点并完成验证。**

## r3 实际发生与根因

在用户明确同意后，使用 `preflight_collector_r3` 对 `live-preflight-scope-r3.json`
（SHA256 `7e7b3d0cdad51228d6b2279324504bb6f13add21dd56202230a4c382bb0804e3`）执行唯一一次连接：

- 两跳认证、`sys.base_prefix` CPython 3.12.13 解析，以及 **三组真实 `validation_remote.py`（`mode="probe"`，含 2 秒超时探针与 64 KiB 输出溢出探针）子进程全部正常执行完毕**，且三次实测的 10 字段 `expected_runtime` 逐字节相等（总耗时 `4.337437417` 秒，本地子进程与进程组干净回收）。
- 随后脚本在末尾汇总断言处返回 `returncode=1`（`stderr=170` 字节，SHA256 `d9700e833c46ee5fce5c2a35158f0217b5bc830cda410c86a649ac7bdb904e50`）：`RuntimeError: probe assertions did not all pass`。
- **确切根因**：核对 `validation_remote.py` 第 124–132 行（`sandbox_argv`），既有后端通过 `--ro-bind` 将四个宿主目录（`/usr/lib`、`/lib`、`/lib64`、`/runtime`）设为只读挂载，而未传入 `--remount-ro /`——`bwrap` 的 `/` 是空的内存 `tmpfs`（仅用于建立隔离挂载点与 `--dir /work`，不含任何宿主目录）。r3 探针中包含了 `open("/probe_root_write", "wb")`，在空内存根 `tmpfs` 上写入成功，从而将 `ro_ok` 置为 `False`。
- 原始 r3 回执完整保留（`caller-return.json` SHA256 `30e6b35a92b394d7129b2463708770dcc853903d6324b8316259fd9d93ef95d6`），未擅自自动重连。

## r4 修正与本地验证

新增目录 `/private/tmp/lora-preflight-r4-20261120-3gi9GH/`（保留全部 r2/r3 原件不变）：

1. `preflight_probe_remote_r4.py`（SHA256 `8a655941e112d27eae52269284aabc4a39346002a7f972e332dd34d520c4ec6c`）：将只读写测试精确对齐 `validation_remote.py` 的全部四个宿主绑定挂载（`/usr/lib`、`/lib`、`/lib64`、`/runtime`），同时核验 `/` 下仅存在预期的 8 个隔离挂载点名称且无任何宿主目录，并将汇总断言细化为按固定检查项名称回报。
2. `live-preflight-scope-r4.json`（SHA256 `ca437b1332afa1a03dd4e03a28a9293e4b1e5391d544087c5867cb17ad99826c`）：安全参数、时限与禁令保持不变，状态为 `DRAFT_NOT_AUTHORIZED_OR_EXECUTED`。
3. `preflight_collector_r4.py`（SHA256 `e2428481857c9079a2a9994e0625d51233956375e85acc25bcd3fdf8eff5cf2d`）及 `test_preflight_collector_r4.py`（SHA256 `afcaa3e5f3f7dde518caa64b6492e8f70953463341063c90ec76fc04124cab60`）：先 RED（4 失败）后 GREEN（4 通过，exit 0）。
