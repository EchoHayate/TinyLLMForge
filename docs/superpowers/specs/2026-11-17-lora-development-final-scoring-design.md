# 开发阶段最终用例与评分绑定

日期：2026-11-17。接续独立最终请求和执行组件；不启动模型或候选。
按用户要求连续推进，不重复要求常规实施确认。

## 选择

选择小型纯函数模块，从原冻结任务材料重建最终用例，再逐条核对真实
执行回执。不能把调用者给出的任意用例列表当成完整评分集；也不能把
旧固定参考函数验证器改成新候选评分器。此模块不引入新测试、修正参考
答案、选择候选或给模型反馈。

复用 `validation_contract._tasks(source_raw, manifest_raw)` 核验已有冻结
来源及元数据绑定，按 D01–D04 身份选定一题。生产调用在可信评分侧读取
已有冻结材料；本次实现和测试只用合成材料，不打开真实私有题库。
协调器以后只能把原公开提示传给生成器，不能传 source bundle 或评分计划。

新增 `development_final_scoring.py`，只包含最终规划及回执归约，不包含
进程、模型、文件写入或开发难度校准。这样用例来源和结果判断可单独审查。

## 接口

`build_final_plan(source_raw, manifest_raw, *, run_id, task_id, arm,
generation_call_id, generation_raw, expected_generation_sha256)`

`score_final(source_raw, manifest_raw, receipts_raw, *, run_id, task_id, arm,
generation_call_id, generation_raw, expected_generation_sha256)`

两者具有相同身份参数。score 必须内部重新构建计划，不接受外部可删行的
计划作为评分真相。全部输入只读；返回值为独立普通 JSON 值。

run_id/generation_call_id 为 32 位小写十六进制，任务仅 D01–D04，
arm 仅 base/text。generation_raw 为最多 1 MiB 的严格 JSON bytes，
SHA256 必须等于 expected_generation_sha256（64 位小写十六进制）。
generation 对象必须是 `LORA_GREEDY_CANDIDATE_UNATTESTED`，其 candidate
为最多 65,536 UTF-8 字节的 str，candidate_sha256 精确匹配候选字节。
保留原 candidate 字节，不 strip、不移除 Markdown、不退回首轮答案。

这只是将候选绑定到指定的生成记录字节，不能认证生成记录的来源。
之后协调器仍须核对 claim/observation、第二轮 message digest、call ID、
模型和 tokenizer 身份及完整运行链。此模块不把 UNATTESTED 改成已认证，
不从调用方提供的 SHA 推断独立可信性。

## 用例集合

先核验输入身份，然后加载冻结任务。选中任务必须为 development，且唯一。
用 `candidate_feedback._valid(task, candidate)` 作与首轮相同的精确签名/
单函数语法判断；语法有效不等于计算政策准入。

最终集合固定为 public_cases 与 scoring_cases 的并集，public 在前，
每组内部按 case ID 排序。public 恰好 P01/P02；scoring 非空、无重复，
case ID 必须符合最终协议且不使用 P01/P02。原 source 中每条 case 恰好
`id/args/expected/reason`（依据 November11 数据计划，不是去掉 reason 的
公开提示投影），reason 为 str 且不进入请求或评分结果，args 为 list。
预期结果沿用候选执行协议的合法
return/ValueError outcome 格式和严格 JSON 规则（规范编码最多16,384字节，
JSON 嵌套最多128层），实际回执的 outcome 同样检查，不作数值类型转换。
这层只使用冻结结果，不执行 reference_source 或 mutant。

case_set_sha256 为如下有序对象的规范 JSON digest：

```
{
  "kind": "LORA_DEVELOPMENT_FINAL_CASE_SET", "version": 1,
  "source_sha256": PINS["task_sources_v1.json"],
  "manifest_sha256": PINS["manifest_v1.json"],
  "task_id": task_id, "task_sha256": digest(task),
  "entry_point": task["entry_point"],
  "cases": [{"case_group": group, "case_id": case["id"],
             "args": case["args"], "expected": case["expected"]}, ...]
}
```

用例数量来自该固定完整集合；本次不假设或宣布实际私有用例数量。
plan 包含 kind=`LORA_DEVELOPMENT_FINAL_PLAN`、version=1、
run/task/arm/generation_call_id/generation_sha256/candidate_sha256、
case_set_sha256、task_sha256、source_sha256、manifest_sha256、
candidate_valid、scheduled_cases、requests 和 expected_outcomes。
expected_outcomes 与完整集合严格同序，只留在可信评分侧。

candidate 无效时 scheduled_cases 仍为完整集合数量，requests 为空；
不能把应评分的分母改成零。有效时每个用例构造最终协议的精确请求；
attempt_id = digest({
`"kind": "LORA_DEVELOPMENT_FINAL_CASE_ATTEMPT", "version": 1,
"run_id": run_id, "task_id": task_id, "arm": arm,
"generation_call_id": generation_call_id, "generation_sha256": generation_sha256,
"case_set_sha256": case_set_sha256, "case_group": group, "case_id": case_id`
})[:32]。要求本计划内唯一。

每条请求经最终 prepare_request 检查；worker payload 仍只有
source/entry_point/args。任何预期值、参考实现、完整计划不得进入 payload。
确定性 attempt_id 只用于记录绑定，不替代执行器的独占 claim。

## 回执归约

receipts_raw 为最多 1 MiB 的严格 JSON bytes，必须为 list。无效候选只允许
空回执，返回 status=`invalid_source`，received_cases=0、passed_cases=0，
scheduled_cases 保持完整分母。

有效候选允许完整回执或严格按计划顺序的前缀，以保留运行中止后的缺失行。
拒绝多行、重复、错序、错 digest、陌生状态、额外字段、非有限值/重复键。
completed 回执恰好 request_sha256/status/outcome，其 outcome 格式沿用
候选执行协议；其他回执恰好 request_sha256/status，status 只能是
program_error/timeout/output_limit。准入拒绝与基础设施异常不能伪造成这些
回执。生产回执来源和外部 lifecycle 重建由后续协调器负责。

每个完整 outcome 用 encoded(actual) == encoded(expected) 判断，严格区分
bool/int/float/字符串。每行 status 为 pass/mismatch/三种执行错误/missing。
全部收到且全部通过才返回 pass；有缺失返回 incomplete；完整但有任意不通过
返回 fail。行不短路，不能在首次失败后删掉后续已收到回执。

结果 kind=`LORA_DEVELOPMENT_FINAL_SCORE`、version=1，绑定 plan_sha256，
保留 plan 的 run/task/arm/generation/candidate/case_set 身份、candidate_valid，
scheduled_cases/received_cases/passed_cases、status、cases（case_group、
case_id、request_sha256 或无效候选时 null、status）。
结果不带 expected_outcomes，也不回传给生成器。

评分结果始终令 FLAGS 全部 false。pass 只是本记录绑定下的功能结果，
不是实验完整性、隔离、难度校准或 LoRA 收益通过。

## 验证范围和后续

本次纯合成 RED/GREEN：冻结加载路径/错 hash、D-only 身份、完整有序集合、
精确来源与候选字节绑定、各元数据变更、无效源码、无源码执行、answer-free
负载、全部/部分回执、类型严格比较、ValueError、错误状态、重排/删行/多行、
重复 JSON 键、编码/预算错误和输入输出别名隔离。
不改旧模块，不读取实际私有数据，不启动候选、远程或模型，不提交/push。

下一协调器预先固定八个 episode 的轮换组序和两轮预算；首轮 invalid_source
仍给规定的第二轮机会，但在开发校准中保留解析问题，即使第二轮修复也不能
把本批标为校准通过。首轮或最终 program_error/output_limit/timeout 分别保留；
最终功能结果与校准标签分开。基础设施失败优先中止，不当作模型纠错反馈。
这些规则需在协调器设计中完整落实，此纯评分组件不代替该阶段。
