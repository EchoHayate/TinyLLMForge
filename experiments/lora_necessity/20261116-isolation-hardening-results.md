# Python 3.12 隔离与权限清空：部分实测结果

日期：2026-11-16，Asia/Shanghai。

状态：**已实测 Python 3.12 隔离启动和全部 capability 清空；
系统调用过滤器已加载，但兼容修正后的完整检查尚未运行成功。**
没有执行模型生成代码，不改变整体候选执行准入标记。

## 现有依赖

20:41:25 的只读查询成功，观察到既有 CPython 3.12.13、
libseccomp 2.3.3 和内核 `cap_last_cap=40`，没有安装新软件。
解释器文件 SHA256：
`7d43f6e86a6c6dd12005ec77eb2055f1be3f1bb3adedf8afe0a87973fa7371ce`；
libseccomp 文件 SHA256：
`010f6a6d48fb08bd76dcd0edd519bb9a8f33d4130fefa43b9106f052db25b762`。

## 本次固定探针方案

不修改共享 bubblewrap，不依赖它的 setuid 权限完成启动：

1. 宿主可信启动器在执行 bwrap 之前设置并核验 `no_new_privs`，
   防止这次 exec 通过 setuid 获得宿主 root 权限。
2. 必须成功创建新的 user/mount/PID/network 等隔离区，失败不得回退。
3. 只在新 user namespace 内短暂保留 `CAP_SETPCAP`，用于丢弃
   bounding set 的第 0–40 位，再清空 ambient、effective、permitted、
   inheritable；核验五组 capability 标志均为零。
4. 把既有 Python 3.12 二进制及标准库只读映射到 `/runtime`，覆盖隐藏
   原有 site-packages；不挂入训练环境、权重、实验数据或账户目录。
5. 使用默认 `EPERM` 的 seccomp 允许列表，只放行固定 Python 探针
   所需调用，保留 socket、fork、execve 和未列入 getppid 的拒绝测试。

所有输入都是审阅过的固定程序，不接受外部候选。
每版实际尝试前经过独立只读审查，源码到远端命令的绑定由主代理复核。
本次限制和探针不恢复全链路内存/磁盘预算或断连自动清理要求。

## 三版尝试：分别保留

### V1：探针的可选参数检查错误

北京时间 20:47:15 连接成功，启动器读取
`/proc/sys/kernel/unprivileged_userns_clone` 时发现文件不存在并退出。
这是把发行版可选参数当作通用必备条件的兼容性错误。
此时还没有启动 bwrap，也未得到权限或过滤器结果。

V2 将检查改为：文件缺失记录 `unavailable`，不据此认定功能支持或禁止；
文件存在但值不是 `1`、或发生其他读取错误仍拒绝。
实际创建 user namespace 和宿主禁提权要求完全保留。
本地四项针对性测试先 RED（缺少实现），后 GREEN 全过。

### V2：启动和权限清空成功，过滤列表过窄

北京时间 20:49:58 连接成功，收到：

```text
USERNS_POLICY_unavailable
HOST_NO_NEW_PRIVILEGES
ALL_CAPABILITIES_ZERO
```

绑定的程序在输出最后一条标记前已经核验：

- Python 为 3.12.13，隔离区解释器前缀是 `/runtime`；
- UID/GID 为 65534，user/mount/network/PID namespace 与宿主不同；
- 宿主禁提权属性仍生效；
- 所有五组 capability 标志均清零。

随后 `seccomp_load` 返回成功，程序进入过滤后的检查阶段，
在 `pathlib.glob -> os.stat` 查询 site-packages 属性时收到 `EPERM`。
没有最终 JSON，不能把前面部分检查算成整个探针通过。
过滤后已计算但尚未输出的检查值，也不能据此认定为 true。
尤其不能从该次结果推断 socket/fork/execve 的行为测试已完成。

### V3：只读兼容修正已审查，启动被认证阻塞

修正版仅增加只读 `stat`、`lstat` 调用及其过滤前后对照，
共预定 17 项检查。旧式文件属性调用是该故障的待验证兼容解释，
尚不能说修正已经在这台机器上通过。
不增加任何网络、创建进程或执行程序调用，不改变默认拒绝策略。

独立审查结论为 `READY_FOR_FIXED_PROBE_V3`。
主代理另行核验远端命令嵌入的启动器/探针字节和摘要。
但实际 SSH 尝试在跳板机认证阶段返回 255，stdout 为空，
没有目标身份确认或探针启动标记。20:54:38 再查本地票据状态仍有效。
这是认证阻塞，不是 V3 代码测试失败；本次没有自动重试。

## 尚不能宣称

- 完整 17 项硬化检查通过；
- 对未知代码的全面隔离安全或整个运行库无漏洞；
- 隐藏答案、收据真实性、候选超时和输出上限已端到端验收；
- 四道开发题或 LoRA 对照已经运行。

下一次先原样验证 V3 的完整结果，必须同时有所有预定检查为 true、
完整 JSON、结束标记和退出码 0；不能只看进程退出成功。
之后仍需独立候选执行器及受信收据接入，再运行原定开发任务。
本次未更新模型、未重跑历史固定任务、未修改旧失败或封存记录。

## 证据摘要

原始记录与含内部路径的启动脚本仅保存在私有阶段目录。

| 记录 | stdout SHA256 |
| --- | --- |
| 依赖只读查询 | `1e2c6176a0718ef08323e675d1891f9ca30ca2a63d9deec028656e63de039a67` |
| V1 | `4378a8bad9df5e58d74c5642abc1c29e47b66f5a2b1bf3005ed682f74b8058e0` |
| V2 | `20ef085cc2d4b51a94c012bffc12d1c9d71c2b07414fd2fe527ae95b35050477` |
| V3 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

没有更换分支、提交或推送；工作仍在 `experiments/feedback-active-verification`。
