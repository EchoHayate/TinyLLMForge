# 远端固定程序隔离试用

日期：2026-11-16，Asia/Shanghai。

结论：**原连接路线恢复成功，固定程序已在既有 bubblewrap 中实际运行。
文件系统和网络等基础隔离有正向行为证据；完整候选执行仍未验收。**
这是固定可信程序试用，不是模型生成代码执行或 LoRA 效果评测。

## 连接情况

用户明确要求继续排查后，对比先前成功/失败日志：
两者使用相同跳板地址、主机密钥和 GSSAPI 认证方式。
不更换账户、票据、认证方式、路由或主机密钥校验。

本次三次调用均完成跳板及目标认证：

| 北京时间 | 内容 | 结果 |
| --- | --- | --- |
| 20:30:20 | 读取 bwrap 身份、选项及解释器版本 | exit 0，约 1.88 秒 |
| 20:31:57 | 固定 Python 隔离探针 | 探针运行，检查未全过，exit 1，约 1.52 秒 |
| 20:32:52 | 对两个失败检查做定向状态观察 | exit 0，约 1.62 秒 |

20:16 的跳板拒绝记录保留在 `20261116-isolation-try-auth-result.md`。
当前证明原配置可以连接，不能据此确定先前为何被拒绝或宣称认证永久修复。

## 固定试跑配置

既有 bubblewrap 0.3.1，文件 SHA256：
`6a43389e0b1a3042b5bf26cc9f55d27cedb536ae5d5b3b2b3879e65d6fad3166`。
使用系统 Python 3.7.3，仅运行预先写好并审阅的探针。

- 明确创建 user、mount、PID、network、IPC、UTS 隔离；
  UID/GID 均为 65534，指定 `--cap-drop ALL` 和独立会话。
- 仅只读挂入 `/usr`、`/bin`、`/lib`、`/lib64`；新建 proc/dev 和临时目录。
  不挂入模型、监督/评分答案、账户目录、Docker socket 或凭据目录。
- 启动前用 `env -i` 设置 PATH 和 LC_ALL；根挂载设为只读。
- 固定程序设 5 秒 alarm、2 秒 CPU 限制、禁止 core；
  只写入再删除一个 16 字节以内的临时文件，只启动一个计算 `6 * 7` 的子进程。
- 没有网络目标连接、漏洞利用、无界 fork、内存压力或生成代码执行。
  未修改服务、安装软件、拉取镜像、替换运行时或接入 GPU。

这些是本次固定探针保护，不恢复被用户取消的全链路内存/磁盘预算，
也不声称验证了断连自动清理或完整候选资源控制。

## 原始检查及进一步解释

第一轮共 12 项检查，10 项为 true：

- UID/GID 符合预期；
- user/mount/network/PID namespace 身份与宿主不同；
- 根及 `/usr` 挂载只读；
- 宿主 `/data00`、`/home`、`/etc`、`/run`、`/var` 不可见；
- 网络设备仅 loopback，无 IPv4 路由；
- `NoNewPrivs=1`；
- 临时文件写读及固定子进程均成功。

两项 false 原样保留，不回写成全过：

1. **环境变量检查过严。** 定向观察只有 `LC_ALL`、`PATH`、`PWD`，
   且 `PWD=/work`。已读取的上游 v0.3.1 源码明确在切换目录后设置 PWD。
   因而该项不是观察到凭据泄漏，而是最初只允许两个变量名的断言不准确。
2. **权限 bounding set 未完全清空。**
   `CapInh/CapPrm/CapEff/CapAmb` 均为 0，`CapBnd=000001c000000000`，
   即第 38、39、40 位仍在。上游源码按编译期 `CAP_LAST_CAP` 循环丢弃权限；
   这与旧构建未覆盖较新 capability 位的解释一致，但本次没有反查该二进制
   的编译头文件，不能把这一解释当成已证实的构建根因。
   bounding 位残留也不等于这些权限当前有效或已经获得宿主权限。

两个进程均观察到 `Seccomp=0`。本次确实没有传入 seccomp 过滤器，
因此没有系统调用过滤验收证据；不是过滤器启用后失效。

## 下一步与边界

这次不是“服务器登不上”或“所有隔离都不可用”，而是已有工具能完成
基础隔离，完整候选运行环境尚缺独立审查和行为验证：

- 可信启动器的 capability/系统调用边界；
- 既有 Python 3.12 运行环境的最小只读投影，不能把系统 3.7 偷换成实验解释器；
- 候选输入、超时、输出上限及受信收据绑定。

之后仍先运行原定 4 个开发任务 base/text 两轮，不能跳到 32 个评测任务。
本次 `generated_candidate_executed=false`、
`candidate_execution_admitted=false`；benchmark、formal、net-benefit、
isolated 等整体就绪标记不变。没有再次训练或评估 LoRA。

## 证据定位

内部路径、账户和 SSH 原始日志只保存在私有阶段目录。

- 准备查询 stdout：`a55279ce1dda33d42705efa1088ba7c51706b5d14bd5eba36861240f81b21d6d`
- 固定探针 stdout：`cef5ad1630df826100f83f7fdc2d29ef97f95afac59fdd5bcb1ffe6a51f3eef9`
- 定向观察 stdout：`74755b54d8ce80e81b9c78d062fa474efd67a5e47452490225f015d7d799c293`
- 上游 bwrap 源码：`a71ce2a6ac0be76a1c12d2f8ba2c480d8eac45109099d1dfbb7d4b94524f2f8f`
- 源码来源：`https://raw.githubusercontent.com/containers/bubblewrap/v0.3.1/bubblewrap.c`

工作仍在 `experiments/feedback-active-verification`，HEAD 为
`c214a1bcb14a6c2c658ab6f9df2bc3bebca50939`。本次只新增结果记录，
未修改已有代码和失败记录，未提交或推送。
