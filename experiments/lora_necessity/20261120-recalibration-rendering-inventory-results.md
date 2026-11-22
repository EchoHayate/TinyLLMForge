# LoRA 校准题库：消息渲染与合成长度清单

日期：2026-11-20，Asia/Shanghai。

本地实现、合成回执测试及两组件集成独立审阅已完成，零项活动发现。
本文件不是 tokenizer 长度结果，也不是 LoRA 效果报告。

## 完成内容

- `recalibration_rendering.py`：为新版 12 道题构造 base/text 首轮消息，
  以及给公开反馈或隐去结果的两种续答。text 保留原有八个监督示例。
- `recalibration_tokenizer_inventory.py`：从固定文件和原始合成回执重新核对
  480 个检查位置，构造并复验 120 行清单；不接受外部只提交一个 PASS。
- 对应两个 `test_recalibration_*.py` 新文件；旧模块和封存数据不修改。

清单保持 24 行首轮、48 行参考答案组合、48 行固定初始程序续答，
不合并相同文本。参考答案是独立字符串，不进入当前任务的 user 消息，
也不手动添加 EOS。反馈仅取重建报告中 initial 的 P01/P02 状态；
不把隐藏用例、私有结果值或错误预测塞入反馈。无反馈分支不接收结果。

缺少完整回执或参考／见证检查失败时，所有类别均不构造：
返回 `DEPENDENCY_UNSATISFIED`、计划 120 行、实际零行。
合成回执完整通过时也只返回 `SYNTHETIC_ONLY`，不是可投递的真实清单。

## 本地验证

从仓库根目录运行，使用已有 Python 和 unittest，没有安装依赖：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_tokenizer_inventory -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_prompt_rendering test_tokenizer_inventory test_recalibration_data_contract test_recalibration_validation test_recalibration_rendering test_recalibration_tokenizer_inventory -v
```

- 渲染器：8 项预期 RED 断言失败，随后 8 项 focused / 60 项当时的窄回归通过。
- 新清单：9 项预期 RED 断言失败，随后 9 项 focused / 74 项窄回归通过。
- GREEN 和回归退出码均为 0，无失败、错误或跳过。
- 74 项已包含新清单 9 项与渲染器 8 项；不同阶段计数不能相加。

测试覆盖严格类型、摘要和原始证据绑定、完整顺序、信息边界、结果容器
独立性、拒绝篡改后重算摘要、数据依赖失败，以及合成标志不能升级。
实际存储的题目程序没有执行，回执由测试人工构造。

主控另用独立 JSON 拼接，逐行重建 120 条消息与参考答案；
行身份逐一对照实施前的显式分叉表，全部匹配。
此检查复用测试构造的合成回执，不是第二份真实执行证据。
八例投影独立重建为 7514 字节，摘要保持不变。

## 身份与证据

证据根目录：`/private/tmp/lora-recalibration-rendering-20261120-JXRgS7`。

| 项目 | SHA256 |
| --- | --- |
| 渲染器 | `ac8286fe3ceb567e23d3a04b32e69d5447d443000c707b3af334cdec94b4baf0` |
| 渲染测试 | `38f91f4759cb5855924006d234c61bb641ba303447758422dd8fe851452f5633` |
| 合成清单组件 | `8328a3f840b3d83fc6525dddc34ba3815eb8cf663e62faad9b965a131ecda8e4` |
| 清单测试 | `8e566526a692d1ad4b26eb5150f57c25f39342d8b62b18ff6e7c9ad4ff83f331` |
| 120 行身份顺序 | `18dffaeabdda4ffa3633d9c3f2e1cb1026c3dd7e2d82cd2de929f9e8ccc769df` |
| 八例投影 | `e1634ad790ed551484a91fe55db36fc2f74b9615d2e5f273989552ea4f6cb290` |

原始 RED/GREEN/回归日志、立即捕获的退出码、实际新增差异与实施报告
分别以 `task-1-*`、`task-2-*` 保存；独立重建结果为
`parent-inventory-verification.json`，核验脚本为 `parent-inventory-check.py`。
173 份基线文件、已完成渲染器及其测试均核对字节未变；
这不是穷尽整个仓库的保护声明。

渲染器已独立审阅通过，0 项 Critical／Important／Minor。
清单及最终两组件集成审阅同样通过，0 项 Critical／Important／Minor。
独立报告为 `task-2-review.md`，SHA256：
`90f2e2a6614e2cb38d4d2ec15aed53c7c2a9ba84041e424968ae4584a0549f97`。
结论限于这两个合成组件，不是整个适配器或分支的准入判定。

最终分支仍为 `experiments/feedback-active-verification`，
HEAD 仍为 `c214a1bcb14a6c2c658ab6f9df2bc3bebca50939`；
tracked/index 无差异，新增文件未提交。主控收尾凭据另存
`parent-final-verification.json`，绑定最终源码、文档、审阅及测试证据摘要。

## 不能据此得出的结论

- 4096 输入 tokens／256 答案含 EOS tokens 只是固定预算元数据，
  不是已经测得的长度；没有调用 tokenizer。
- 真实来源、真实动态完成、长度通过、tokenizer 投递、运行授权及
  benchmark／formal evaluation／net benefit／通用隔离标志全部 false。
- 固定续答用题库 initial_source；不证明将来任意模型续答都能装入预算。
- 没有新模型输出、正确率或 LoRA 增益。本阶段没有 SSH、网络、题目执行、
  模型加载、生成、训练、安装、遥测、提交／推送、分支操作或清理旧文件。

下一步是新版 tokenizer 测量回执报告和会话协调组件的本地实现；
真实清单的来源核验、具体隔离启动包与实际执行仍需独立完成。
这次只完成消息渲染与合成清单，不宣称整个适配器或分支已经就绪。
