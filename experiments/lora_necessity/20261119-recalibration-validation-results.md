# LoRA 校准题库：排程与回执组件结果

日期：2026-11-19，Asia/Shanghai。

本次完成代码实现、本地合成回执测试及独立复审，活动阻断项为零。
这不是题目真实执行通过，也不是 LoRA 效果验证。

## 这一步做了什么

为新版 12 道题整理固定检查顺序：每题 5 份程序，每份 8 个用例，
总计 480 个位置，分成 15 批，每批 32 个。这里没有运行这些程序。

新增组件从封存文件重新构造身份和封包，再根据留存的原始回执重建结果。
它区分三种情况：

- 已观察到结果，包括正常答错以及明确的进程失败。
- 根据提供的采集记录已经投递，但没有可信结果：记为未知。
- 后续批次没有投递记录：在该份采集历史中记为未投递。

这些状态描述传入的采集历史，不独立证明远端到底发生过什么。
身份或回执格式不可信时，整批结果不参与通过判定；先前可信批次仍保留。
发现执行或证据异常后，不接受继续追加的批次，不自动重试。
正常答错则继续统计，不误当成执行器故障。

## 文件与接口

- `experiments/lora_necessity/recalibration_validation.py`
  - `build_schedule(raw_files)`：固定 480 行及逐行摘要。
  - `build_batches(raw_files, worker_raw, run_ids)`：15 个有大小上限的封包。
  - `build_report(raw_files, worker_raw, run_ids, records, expected_runtime=...)`：
    纯回执解析、身份绑定、全分母结果和停止原因。
- `experiments/lora_necessity/test_recalibration_validation.py`：
  本地合成回执和静态投影测试。
- 实施计划：
  `docs/superpowers/plans/2026-11-19-lora-recalibration-validation.md`。

每个 worker 请求只有 request_id/source/entry_point/args，不带正确答案、
见证预测、程序角色或整份私有题库。新模块不启动进程，不连接远程，
不导入 worker 或远程执行器；旧执行器的 2 秒／90 秒期限没有改动。

## 已观察到的本地测试

工作目录为仓库根目录，使用已有 Python unittest，没有安装或上传。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_validation -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_validation_contract test_prompt_rendering test_recalibration_data_contract test_recalibration_validation -v
```

- 修复后新模块：27 项测试通过，退出码 0，无跳过。
- 修复后窄回归：57 项测试通过，退出码 0，无跳过；其中已经包含上述 27 项。
- 初始实现前有 15 项预期断言失败，扩展后有 25 项预期失败。
- 自查另发现一种严格类型问题：Python 字符串子类可绕过旧字段集合
  的等值检查。新增测试先出现 3 个失败子用例，再在新模块中补上
  内建字符串键检查；没有修改旧解析器。

测试包括固定顺序、最小投影、大小上限、布尔值与整数不混同、参考不符、
见证预测不符、超时、缺失结束记录、乱序／重复／多余回执、身份和运行环境
漂移、停止后追加记录、后续批次未知，以及输入／结果容器不互相改写。
这些结果来自合成 JSON 回执，不是执行题目所得的正确率。

主控另从原始 JSON 独立投影封包，与实现生成的 15 个摘要逐一匹配。
当前封存输入的最大源码／请求／worker／整包大小分别是
735／976／1700／24252 字节，均低于原有上限。

## 判断含义

- `matrix_complete`：480 个位置都有完整、有效、正常完成的回执。
- `data_checks_satisfied`：除完整性外，192 个参考位置全部符合正确答案，
  36 个指定错误程序见证全部符合预写预测且区别于正确答案。
- 合成回执可以使上述纯判断为真，但 `execution_provenance_verified`、
  `full_dynamic_validation_complete`、`run_authorized` 和所有准入标志
  仍固定为 false，不能据此声称真实执行、隔离、训练或收益已经通过。

报告包含原始回执和私有结果，只作本地私有证据，不是公开发布材料。

## 身份与保护

证据目录：`/private/tmp/lora-recalibration-validation-20261119-OXqFOy`。

| 项目 | SHA256 |
| --- | --- |
| 新组件 | `a65fe7813b9ffd9db20dd232350737a528716d03583255fa907706be6f9076ad` |
| 修复后新测试 | `cd44a8f45ae12ebb1b2abd4b6643b1b1267a8d6ecf0274cdd0fcfe331cce5dc3` |

主控已核对本轮基线中的 170 份既有 LoRA 文件字节未变。
这是有明确清单的保护检查，不是声称穷尽全仓所有文件。
源码、原始日志、退出码、差异、独立投影及摘要分别见证据目录中的
`task-1-report.md`、`task-1-*.log`、`task-1-*.exit`、`task-1.diff`、
`parent-static-projection.json`、`parent-integration-check.json`。
最终收尾凭据为 `parent-final-verification.json`：170 份基线文件未变，
15 个包摘要再次匹配，最终源码／测试／报告摘要一致；空采集历史保留
480 个未投递位置，不产生任何完成或准入通过。分支与 HEAD 未变，
tracked/index 无差异；新增文件保留在当前工作区，未提交。

独立复审：Spec Compliance Compliant / Task Quality Approved；
0 项活动 Critical／Important／Minor。本子计划通过审阅，不代表完整
适配器、整个分支或后续真实运行已经验收。

最终实施报告 SHA256：
`46e25ce8fe48d3acfeca66ef6e85da5e0d843c887bec057c5a4d4e5e7dea89d7`。
最终独立审阅 SHA256：
`0b407fda8796acf4144566f3d901291c989234f95922a59ec3dfc0d0dfe37e2d`。
二者保留原始发现、修复和复审历史，最终结论在各文件末尾。

初审确认的两项测试问题已修复：运行 ID 的错误用例被另一项无效输入
提前拦截，以及布尔／整数测试实际上只比较了一个布尔值和一个列表。
旧的 26／56 通过计数不能证明这两个断言足够有效，保留为修复前历史。
现在分别用有效 worker，以及形状完全相同、只把布尔叶子换成整数的
参考／见证结果验证，并保留了故意绕过校验或改用宽松比较时的真实失败记录。
试验后实现文件恢复至同一 SHA256；最终 27／57 项通过来自恢复后的正确代码。
复审依据新差异和新原始日志确认修复，不覆盖原始报告。

另有一项返回值格式意见，经审阅者核对实际后端后撤回：
固定后端会先将不支持的返回值标为 `program_error`，再发出回执；
组件保留这种明确失败。它不会把任意可 JSON 序列化的值都当成有效
`completed` 回执。原协议的有界 ASCII 结果格式保持不变，没有为此
放宽旧执行器或解析器。正常答错继续统计，是指该协议内的有效完成结果。

## 尚未完成

本阶段没有 SSH、题目真实执行、tokenizer 或模型加载、生成、LoRA 训练，
没有安装、遥测、提交、推送、分支切换或清理旧文件。封存题库的
dynamic/tokenizer pending 状态不变。

下一组件是公开消息渲染与 120 行长度清单适配。那是后续工作，
本文件不把它记为已实现，更不代表真实 tokenizer 已测量通过。
后续真实执行仍需要具体启动包和可信采集流程审阅。
