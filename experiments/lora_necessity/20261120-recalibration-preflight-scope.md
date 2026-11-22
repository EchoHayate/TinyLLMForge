# 新校准：实时隔离与运行时预检草案范围

日期：2026-11-20，Asia/Shanghai。

状态：**r4 单次实时隔离与运行时预检已执行并通过全部 13 项探针检查（见 `20261120-live-preflight-r4-and-480-package.md`）；15 批 480 项完整启动包已本地构造就绪。**

## 背景与目的

单次只读路线查询（`20261120-route-observation-results.md`）已确认当前两跳严格认证路线连通、目标身份一致且 `/usr/bin/bwrap` 存在，同时确认默认精简 `PATH`（`/usr/bin:/bin`）下不包含 Python 3.12。

按照 `2026-11-20-lora-recalibration-fixed-execution-design.md` 第 3.2 节，在派发任何固定源码批次之前，必须针对目标机上既有的显式 Python 3.12 环境根目录与 `/usr/bin/bwrap`、`libseccomp` 执行一次人工固定程序隔离预检，并采集与远端监督子进程 **完全相同文件描述符（FD）布局** 下的 10 字段 `expected_runtime`。

## 冻结草案身份与边界

- 私有命令绑定范围文件（r4，完整保留草案、r1、r2、r3 及其回执与审阅原件）：
  `/private/tmp/lora-preflight-r4-20261120-3gi9GH/live-preflight-scope-r4.json`
- r4 SHA256：
  `ca437b1332afa1a03dd4e03a28a9293e4b1e5391d544087c5867cb17ad99826c`
- 固定远端探针源码（`preflight_probe_remote_r4.py`）SHA256：
  `8a655941e112d27eae52269284aabc4a39346002a7f972e332dd34d520c4ec6c`

本范围严格限定为**单次连接链路**（工作至多 45 秒、本地回收至多 2 秒；stdout 至多 64 KiB、stderr 至多 128 KiB），复用已验证的两跳严格主机密钥/GSSAPI-only/禁转发/禁复用/禁自动重试安全参数及本地 6 份配置与 `Include` 闭包校验。

拟执行内容仅包含人工固定探针（`mode="probe"`，不含任何题库源码、答案或见证）：

1. 核对显式候选 Python 3.12 可执行文件存在且位于该环境根目录内（缺失直接失败，不搜索其他目录或回退系统 Python）。
2. 采用与 `recalibration_execution_remote.py` 完全相同的干净子进程启动方式（`close_fds=True`、`pass_fds=()`、在 `exec` 前关闭本地配置文件），实测并输出完整 10 字段 `expected_runtime`（含 `seccomp` 过滤器对应的真实 FD 编号）。
3. 在 `bwrap` + `seccomp` 内验证：`mnt/pid/net/user` 命名空间与宿主不同、`NoNewPrivs=1` 且 `Seccomp=2`、根与运行库挂载只读、宿主目录及外层合成假文件不可见、`socket`/`fork`/`clone3` 系统调用均返回 `EPERM`。
4. 验证 2 秒超时终止与直接子进程回收，以及 64 KiB 输出上限截断拒绝与直接子进程回收。

**明确禁止**：不上传或执行 12 题题库源码与评分数据，不加载 tokenizer 或预训练模型，不查询或占用 GPU，不进行模型生成或 LoRA 训练，不探测真实凭据文件，不安装依赖或修改远端配置，失败不自动重试。
