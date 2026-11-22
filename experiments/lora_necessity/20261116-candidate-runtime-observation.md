# 候选执行环境：2026-11-16 只读复核

查询时间：2026-11-16 11:46:42 UTC（北京时间 19:46:42）。
查询退出码 0，目标身份及结束标记匹配，耗时约 2.26 秒。
本次不是认证失败，没有创建容器或执行候选代码。

## 当前观察

| 项目 | 只读输出 |
| --- | --- |
| Docker Engine | 24.0.9 |
| containerd | 1.7.13 |
| Docker 报告的 runc | 1.1.7，commit `v1.1.7-0-g860f061b` |
| Docker 默认 runtime | runc |
| Docker 列出的 runtimes | runc、io.containerd.runc.v2、nvidia |
| 主 runc 文件 | 1.1.7，SHA256 `c3aadb419e5872af49504b6de894055251d2e685fddddb981a79703e7f895cbd` |
| 另一个 runc 文件 | 1.1.12，SHA256 `9ba15f76f011e79e781ae2db619a04489094dd05e0fe824188357f55207f4234` |
| bubblewrap | 0.3.1，setuid 模式 |

命令 PATH 中未找到 runsc、kata-runtime、crun、nsjail、podman、
firecracker、qemu-system-x86_64。这不是全盘搜索，也不能证明机器上没有
任何其他执行环境。替代 runc 文件存在，不证明 Docker 正在使用它；
metadata 与文件摘要也不等于进程级路径绑定或隔离强制性测试。

## 判断

旧的默认 runtime 版本问题仍未消失，候选执行维持未准入。
官方 runc 安全公告 `GHSA-xr7r-f8xq-vfvv` / `CVE-2024-21626` 的
GitHub API 在本次成功读取，列出的受影响范围为
`>=v1.0.0-rc93,<=1.1.11`，该问题修复版本为 1.1.12。
公告也列出与可信镜像、工作目录及 exec 使用方式有关的缓解条件；
本次没有验证这些条件，不声称“任意启动必然逃逸”。
修复这一个公告也不等于截至今日所有漏洞、内核和隔离配置均合格，
因此不直接把另一个 1.1.12 二进制当成可用的安全后端。

本阶段不替换 runtime、不修改 daemon、不安装软件或自行扩权。
后续要么由环境负责人提供已维护的独立隔离后端，要么在另行允许的范围内
完成后端部署及审查，再验证文件系统、网络、凭据、进程及执行时限边界。
这与已取消的总内存/磁盘预算、断连自动清理要求无关，不重新引入这些门槛。

## 证据与范围

- 只读命令 SHA256：`859b472ca070ba489977a16a0cb1369fd2bddbd6515d01f981ce96fc517df739`
- stdout SHA256：`0b0abc65f2169c2986db6db4351017d20a6b6606e94ea25bcc304145cbc1f0ad`
- stderr SHA256：`c60fba12d9210ea65cd5c3d9a80ac3d8a6397b75e20ba9b103fa6d9ddd883729`
- 官方公告 JSON SHA256：`331b06923b6f5cebea1bf6e0efa7a4ac9546a6edd0a52ed4aab5143dab0fae30`

原始记录只留在私有阶段目录；不把账户、连接路由或内部文件路径写进公开报告。
本次观察不推翻已经完成的真实训练和 greedy 组件验证，但不能支持
“候选已安全执行”“已完成两轮修复”或“LoRA 已产生收益”的结论。
