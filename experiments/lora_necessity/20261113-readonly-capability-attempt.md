# 只读资源能力核查：首次调用失败

日期：2026-11-13，Asia/Shanghai。

状态：**LOCAL_PROXY_COMMAND_FAILED_BEFORE_REMOTE_QUERY**。
已执行获准的一次 SSH 调用，但没有取得远端身份或资源能力数据。
不能据此报告服务器不可用、认证失败或资源后端不满足要求。

## 授权与本地检查

用户在只读能力核查范围确认后回复“继续”，授权目标仅为既有
`rl-server`，一次连接尝试，不自动重试。没有授权运行模型、训练、
创建容器、安装、配置资源限制或传输私有数据。

本地核对了目标和跳板路由、已保存主机密钥，以及既有 FILE 缓存的
`klist -s` 返回值（0）。没有读取票据内容。
这证明当时本地有效性检查通过，不证明服务器会接受该认证。

发现默认配置存在 agent 转发和非严格 host-key 校验，因此本次命令
显式使用不读取默认配置的模式、已保存的 known_hosts，以及：

- 禁止 agent 转发、凭据委托、密码交互和连接复用。
- 严格主机密钥校验，不写入新密钥、不修改配置。
- 只有一个指定目标和既有跳板，不选择其他会话或主机。

原始命令、详细路由及输出保存在权限受限的本地证据目录：
`/tmp/tlf-tokenizer-capability-20261113-jlIeeJ`。

## 实际结果与根因

唯一一次调用返回 SSH exit 255，用时约 0.024 秒；stdout 为 0 字节，
stderr 为 80 字节。没有身份匹配或核查完成标记。
本地 SSH 进程已回收，不据此声明远端清理或整棵进程树清理已验证。

直接错误：

```text
/bin/sh: line 0: exec: exec: not found
Connection closed by UNKNOWN port 65535
```

这是本次操作脚本的错误：我在 ProxyCommand 开头多写了 `exec`。
本机 OpenSSH 会自行添加 shell 的 `exec`，实际形成 `exec exec ...`，
因此本地 shell 未能启动跳板 SSH。`UNKNOWN port 65535` 是该失败的
后续诊断，不是已确认的远端端口、主机拒绝或 Kerberos 错误。

依据：

- 本机 `/usr/share/man/man5/ssh_config.5` 的 ProxyCommand 说明。
- 两次**不联网的本地复现**，代理程序仅为 `/usr/bin/true`：
  错误写法打印 `Executing proxy command: exec exec /usr/bin/true`；
  正确写法打印 `Executing proxy command: exec /usr/bin/true`。
  两者都不会建立目标网络连接，也都不可能完成 SSH 会话。
  后者的 exit 255 是无 SSH 服务端的预期结果，不算连接成功。

## 修正及验证

- 原失败脚本完整保留为 `attempt-1-probe.py`，其 SHA-256 与
  首次 `scope.json` 的 helper 摘要一致；原始 scope、receipt 和
  stdout/stderr 不覆盖。
- `probe.py` 仅删除 ProxyCommand 中多余的 `exec `。
- `check_proxy.py` 只解析并执行脚本的导入/常量头部，不运行连接入口。
  修正前 `proxy-red.log` 记录断言失败，修正后 `proxy-green.log`
  记录通过，同时检查两层 SSH 的禁用转发/委托、严格主机校验及
  单次连接设置。
- 没有运行修正后的真实连接，没有上传任何项目源码或输入材料。
  当前根已有独占 scope/receipt，不能把它重新用作一次新尝试的输出根。

原始 receipt 的 `failure=null` 仅表示本地包装脚本没有抛出异常；
SSH 的 `exit_code=255` 和缺失完成标记明确表示本次核查失败。
不能把包装脚本自身的 exit 0 解释为 SSH 或核查成功。

## 后续入口

远端资源能力核查仍未完成。修正后如获准再次尝试，应创建全新私有
证据根，保持同一只读范围、安全选项和单次预算；不得覆盖本次失败。
这不是重新选择后端、启动工作负载或放宽原验收标准的授权。

本次未增加 LoRA 效果证据，未重跑已封存的 314 项组件回归，
未更改其源码、数据或封存结果，未提交或推送。
