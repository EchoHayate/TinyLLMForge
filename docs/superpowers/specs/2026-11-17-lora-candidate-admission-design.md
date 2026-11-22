# 真实候选的保守计算语法准入

日期：2026-11-17。接续用户授权的连续实施，不改写或放宽冻结评测。

## 选择

任意 Python 可以反射或进入 native 代码；固定篡改样例通过不证明包装器
能抵御这类行为。选择在现有格式校验之后增加独立、默认拒绝的计算语法准入，
而不是继续追逐所有解释器篡改方式，或另造一套解释器改变 Python 语义。
仍执行原文，仍使用原 Python、隔离方式和评分器。

准入拒绝是基础设施/执行政策不兼容：停止整个当前协议，不产生
invalid_source/program_error/timeout 反馈，不占用一次“纠错重试”，
不把未运行的候选计作实际执行或算法失败。报告协议不完整，不得由此
比较 LoRA 胜负。规则在任何真实候选生成之前固定；若以后根据输出扩展
规则，必须登记协议变化，不悄悄重跑当前实验。

## 闭合语法

一个无默认值、注解、装饰器的普通函数，使用内置 JSON 参数。
允许返回、赋值/增量赋值、表达式、if/for/while、break/continue/pass、
assert、raise、try/except/else/finally、列表/字典/集合与推导式、
生成器表达式、切片、运算/比较/布尔表达式和条件表达式。
允许 f-string 的值表达式，但其内部表达式仍逐节点检查。
未明确列入的 AST 节点拒绝，包括 import、lambda、class、嵌套函数、
with、yield、async、global/nonlocal、match、具名表达式和星号解包。

名称必须是参数、函数内绑定、当前函数或明确允许的内置名称，
且不能以双下划线开头。调用目标只能是允许的内置函数、当前函数递归调用、
或下列方法；不允许通过变量/下标/调用结果间接调用。
政策 v2 按实际函数/推导式作用域解析名称：推导式目标不能授权外层同名读取，
推导式最外层 iterable 在其外层作用域中检查，其余表达式在推导式作用域中检查。
所有允许的内置调用名称和当前入口名都禁止在任何作用域重新绑定，
包括参数、赋值、循环、推导式目标及异常目标；即使绑定没有被调用也拒绝。
入口名不能与允许的内置调用名称相同。普通非保留局部变量和合法推导式仍允许。
属性只能在允许的方法调用位置出现，不允许属性赋值、单独取出方法、
访问 dunder/frame/globals 或其他数据属性。

内置调用：abs/all/any/bool/dict/enumerate/filter/float/int/len/list/map/
max/min/range/reversed/round/set/sorted/str/sum/tuple/zip，以及
ValueError/TypeError/RuntimeError/ZeroDivisionError/Exception。
不允许 open、eval、exec、compile、getattr、setattr、type、vars、
globals、locals、__import__、input、print 或构造自定义类型。

允许方法：append/extend/pop/insert/remove/clear/copy/reverse/sort/
count/index/get/keys/values/items/update/setdefault/add/discard/
union/intersection/difference/issubset/issuperset/
lower/upper/casefold/capitalize/title/strip/lstrip/rstrip/split/rsplit/
splitlines/join/replace/find/rfind/startswith/endswith/isdigit/isalpha/
isalnum/isspace/islower/isupper/ljust/rjust/center/zfill/removeprefix/removesuffix。
不允许 format、format_map、encode/decode 或字符串指定的动态属性访问。

这些限制依赖输入确实只有 JSON 内置类型，且不能导入或构造新类；
不得把有自定义方法的宿主对象作为参数传入。
在常规请求校验前先检查精确内置类型，拒绝容器、键和标量的子类，
不触发其自定义比较、迭代或序列化钩子。
白名单不是替代系统隔离、CPU/时间与输出约束，也不声称解释器无漏洞。

## 接口与接入

新增 `candidate_execution_admission.py`，不改 `candidate_valid` 的冻结
格式判定。`admit_request(request)` 复用现有严格请求校验，再进行完整 AST
遍历，成功返回独立政策记录（固定政策版本、完整请求 SHA256）。
失败抛 `CandidateAdmissionError`（基础设施错误），不执行任何源码。

生成候选使用新增 `run_candidate_request`，先对请求做不可共享的 JSON
快照，再准入，成功后才进入既有 `run_request`。准入记录单独保存，
不污染现有收据格式。固定安全诊断仍可直接使用低层 `run_request`，
但它不提供生成候选准入。

本地测试只做语法检查和合成控制流检查。覆盖普通数值/字符串/循环函数，
全部禁止路径、伪装名字、属性取值/赋值、间接调用、异常目标，
以及准入失败必须发生在进程启动和 claim 之前。
独立审查必须检查 AST 白名单闭合性和宿主对象注入，不仅看测试数量。

## 本轮身份与验收

政策版本为 2，政策描述摘要为
`16fb67e48d8adba83ad0cc832053d65d2b4d71a6b0075407d1c4a0e006410859`。
四字段记录为 kind/version/policy_sha256/request_sha256；同时固定实际校验器源码，
政策描述摘要本身不是正确性证明。此版本变更发生在任何真实候选生成之前，
用于修复独立审查指出的作用域漏洞，不是根据模型答案扩展允许语法。

准入记录必须以独占方式、0600 权限完整写入并关闭后才启动控制端；
写入不足、写入或关闭异常均中止。后续控制端错误保留该记录而不重试。
新远端固定样例批次独立于已完成 V2：四个允许的 toy 执行、三个静态拒绝，
以及三个低层“先输出结果再超时/超限/非零退出”诊断。低层诊断不经过真实
候选准入，不能把其成功启动解释成允许候选使用同样的能力。
