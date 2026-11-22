# 开发阶段最终候选准入与执行

日期：2026-11-17。仅接续独立最终请求协议，不启动真实任务。

## 目标与选择

给 D01–D04、base/text 的第二轮候选建立真实的最终执行域。
不把最终请求伪装成 round=1，不改变已封存首轮协议，也不在运行时
替换旧模块的全局 `prepare_request`。

新建两个小型版本化组件：

- `development_final_admission.py`：相同 computation policy v2，验证新请求、
  固定快照、保存新域准入记录，然后才调用最终控制器。
- `development_final_controller.py`：原控制器的独立版本；仅更换
  `prepare_request` 的导入为 `development_final_protocol`。

暂时保留控制器主体的明确副本，便于核对其与封存版本只有协议导入差异。
这比动态执行/改写模块源码或伪造公开请求更直观。后续可以单独讨论抽取公共
核心，但本阶段不以重构改变旧证据。worker、READY/result/lifecycle 规则不变。

## 准入

`admit_request(request)` 先用原生 JSON 类型遍历和新的 `prepare_request`
拒绝非法输入；接着执行与 `candidate_execution_admission.admit_request`
完全相同的 AST、调用、属性、词法作用域和保留名字检查。

直接复用原模块的政策常量和纯词法辅助函数。不调用旧 `admit_request` 或旧
`prepare_request` 来制造首轮检查记录。新域不增加任何 AST 节点、builtin、
方法、反射、文件、网络或进程能力。

通过时返回且仅返回：

```
kind = DEVELOPMENT_FINAL_AST_ADMISSION
version = 2
policy_sha256 = 原 computation policy v2 摘要
request_sha256 = 完整最终请求摘要
```

`run_candidate_request(request, argv, *, artifact_dir, cwd, env, **limits)`
保留原接口形状，但只接受最终请求：

1. 原生 JSON 类型检查、新协议校验、规范编码后重新解析为独立快照。
2. 对快照准入，0600/O_EXCL 保存 `admission.json`，完整写入并关闭成功后
   才允许启动控制器。fdopen 失败时关闭已打开描述符。
3. 调用新控制器；核对最终 receipt 的 request_sha256 与准入摘要一致。
4. 返回控制器原结果及独立 `admission` 字段。

拒绝/保存失败属于基础设施或协议中止，不是模型错误、反馈、免费重试。
继续采用 `CandidateAdmissionError`（`InfrastructureError` 子类）。
写入失败不删掉已占用的准入文件以制造重试机会。

## 控制器

运行与结果接口沿用 `candidate_execution_controller.run_request`。
完整实现必须逐字节等于封存控制器将以下导入替换后的文本：

```python
from candidate_execution_protocol import (
    InfrastructureError, READY, make_receipt, prepare_request,
)
```

替换为：

```python
from candidate_execution_protocol import InfrastructureError, READY, make_receipt
from development_final_protocol import prepare_request
```

原控制器 SHA256：
`a9c72892b97eb4cb69492cabd53cff3be2822a4837291fc59709d0be61ee4bc0`。
测试必须核对上述确切关系；不能借新域名修改生命周期、资源上限或返回语义。
worker 只能收到单例 `source/entry_point/args`，不接收预期答案、完整用例集、
其他候选、生成记录或宿主控制信息。

## 本地验证范围

只使用合成请求和预先写好的可信控制帧 helper。helper 不解释或执行请求内的
source；不运行真实生成的候选，不使用远程或真实模型，不读取私有题库。

测试至少覆盖：

- 同政策普通计算通过，反射/文件/进程/间接调用/作用域绕过被拒绝。
- 原始对象修改不改变传给控制器的准入快照。
- 准入拒绝、短写、写入/关闭/fdopen 失败时不启动控制器；重复准入不覆盖。
- 回执摘要不匹配中止；新旧域不能重放；只输出三字段 worker payload。
- READY 前不发 payload；正常/ValueError 结果、重复/缺失/伪造 stdout 结果、
  非零退出、先发结果后挂起/洪泛、启动失败、超时和 kill/reap。
- 字节上限、独占 claim、元数据与保存的原始回执一致。

可以通过继承原控制器 TestCase 并仅替换 module/request 复用生命周期测试；
须检查所有被继承方法的假设，不能把原有负例变成新域合法输入却仍声称覆盖。
原 TestCase 通过模块属性引用，避免无意再次发现同一原测试类。

## 边界

通过只证明最终请求域复用同样的语法政策和进程控制；不是独立评分结果。
真正运行仍需要新运行身份/固定 launcher、实际隔离检查，以及可信协调器对
第二轮生成、冻结完整用例、逐例输入与回执的绑定。最终通过要求全部公开与
独立评分用例的并集通过，评分不能回传模型。不得重放旧诊断或启动 32 题。
