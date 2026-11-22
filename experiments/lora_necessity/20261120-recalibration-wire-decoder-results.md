# LoRA 新校准：执行记录五帧解码结果

日期：2026-11-20，Asia/Shanghai。

状态：五帧字节解码子计划通过实现、主控独立检查及新审阅者检查。
这只表示字节层可用，不表示完整执行证据重建或真实运行已完成。

## 本轮完成的能力

新增 `recalibration_execution_wire.py` 和
`test_recalibration_execution_wire.py`：

- `decode_frame` 读取一帧，允许未来控制器先检查单独收到的 READY 帧，
  无须等到任务结束。返回值仍只是原始 payload bytes。
- `decode_wire` 核对原始字节摘要，再按固定顺序读取 READY、后台 stdout、
  后台 stderr、诊断、DONE 五帧。
- 拒绝错误类型、magic、通道、截断、重复、额外字节、超限长度及摘要不符；
  不跳过诊断，不修剪字节，不把残缺前缀当成完整记录。
- 全部帧头和内容合计最多 8 MiB。后台 stdout 单帧的上限也是 8 MiB，
  但它占满后加上帧头就不能成为合法完整 wire；两种边界分别检查。

这是既有设计第 4.3 节的字节层，不是完整的执行证据重建器。
READY/DONE 的 payload 在这一层保持不透明：能读出 bytes 不表示它们是
合法 JSON、符合消息 schema、属于指定批次或证明进程已结束。
摘要匹配也只表示与调用者提供的摘要一致，不证明真实来源。

## 检查结果

私有证据根：`/private/tmp/lora-recalibration-wire-20261120-lrR7Iy`。

| 检查 | 已观察结果 |
| --- | --- |
| 实现前 RED | 31 项预期断言失败，exit 1 |
| 聚焦 GREEN | 31 项通过，exit 0 |
| 窄回归 | 49 项通过，exit 0，包含上述 31 项及已有 18 项 package 测试 |
| 主控独立检查 | 五帧字节全部一致，8 种损坏/超限输入拒绝 |
| 边界 | 含 70 字节帧头共恰好 8 MiB 通过，多 1 字节拒绝 |
| 摘要顺序 | 原始摘要匹配检查先于帧头解码 |
| 保全 | 187 份既有文件全部未变 |
| 独立审阅 | 规格通过、代码通过，P0/P1/P2 问题均为 0 |

聚焦与回归均无跳过或错误；原始日志及即时退出码保留在
`task-1-{red,green,regression}.{log,exit}`。主控读取日志并另运行
`parent-check.py`，保存 `parent-verification.json`。
没有重跑已完成的同 hash suite 或此前 66/157 项 suite。
窄回归是同一解释器下的组件兼容检查，不是启动链集成测试。
新审阅者核对实际 470 行新增 diff、源码/测试身份及保存的原始证据，
未复跑测试。`fresh-review.md`、`report.html`、`report.md` 保存审阅结果。
`final-verification.json` 绑定最终代码、日志、审阅和本文摘要；其中
ledger 摘要仅对应封存时版本，后续追加不改写旧回执。

源码 SHA256：

- `recalibration_execution_wire.py`：
  `4b2bfb55b1f56cf8bd41d2f70cee970131715ae0268cfa4e0cc83ccccf1ad722`
- `test_recalibration_execution_wire.py`：
  `00353b50e3e7e2f47d819771a0bb46fdf583224d9492c8305344856644fb3eb7`

## 还没有完成的部分

后续需冻结并检查 READY/DONE 的精确字段、批次/配置身份和进程结束
回执，再将后台原始输出接回已有 480 项内容报告。后台进程结束与
本地 SSH 结束必须分开；仅有退出码 0 或读到 32 行不能升级为成功。
远端监督、本地主控、完整源码/启动配置、实时预检、一次性范围以及
受信真实采集接受仍未完成。

本轮没有真实执行题目、SSH/网络、加载 tokenizer/model、生成或训练。
没有新的 LoRA 收益数据，也没有改变冻结语料、评分或 token 预算。
没有安装、遥测、清理、切分支/worktree、commit/push。
