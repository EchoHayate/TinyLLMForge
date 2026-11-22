# LoRA 新校准：固定请求准备组件结果

日期：2026-11-20，Asia/Shanghai。

结论：本地准备子计划通过实现、独立重建核验及新审阅者检查。
现在可以确定待测的 480 项请求如何分成 15 包，并检测输入、身份或
包内容被改动；不能据此启动远端，也没有新增 LoRA 效果数据。

## 本轮交付

- `recalibration_execution_package.py`：纯准备构造器和重建校验器。
- `test_recalibration_execution_package.py`：18 项聚焦测试。
- 输出固定为 `PREPARATION_ONLY`，所有启动、授权、来源、动态检查及
  研究结论标志均为 false；public 保存身份摘要，private 保存精确包字节。
- 校验从原始输入重新构造整个对象，而非仅检查调用者提供的摘要。
  修改标志后重新计算摘要不能获得准入。
- 六份依赖源码只是已冻结的子集，不是完整执行源码清单，更不证明
  当前解释器实际加载的源码身份。后台执行器和 worker 仅按字节校验。

## 验证与审阅

私有证据根：`/private/tmp/lora-recalibration-package-20261120-2GDyzS`。

| 检查 | 结果 |
| --- | --- |
| 实现前 RED | 18 项断言失败，exit 1，无导入错误 |
| 聚焦 GREEN | 18 项通过，exit 0，无跳过 |
| 窄回归 | 66 项通过，exit 0，无跳过；已包含上述 18 项 |
| 主控独立原始 JSON 投影 | 480 项请求身份和 15 包字节身份全部匹配 |
| 重新摘要后的伪造准入标志 | 主控另验三种，全部拒绝 |
| 既有材料保全 | 184 份基线文件全部未变 |
| 新审阅者规格与代码检查 | PASS，P0/P1/P2 均为 0 |

实现者保存原始日志及即时退出码；主控读取这些日志并执行独立核验。
新审阅者检查实际新增文件 diff、源码、直接依赖、日志和主控回执。
未重复运行相同 hash 的已完成 suite，也未重跑旧 157 项 tokenizer suite。
这些测试只检查准备行为，不执行存储题目，不替代真实动态验证。

最终源码 SHA256：

- `recalibration_execution_package.py`：
  `1feb26e08f62afdd1895b9cb32ffa3884a26ebada546ba807c931783a6382d90`
- `test_recalibration_execution_package.py`：
  `61982a6fc5452f272c6eeaf69b97a0e951c8332c7c10aed0bad4bde28d66aaf1`

证据文件：

- `task-1-report.md` 与 `task-1-{red,green,regression}.{log,exit}`
- `task-1-added.diff`
- `parent-projection-expectations.json`、`parent-verification.json`
- `fresh-review.md`、`final_comments.json`、`report.html`、`report.md`
- `parent-final-verification.json`：最终源码、报告和文档摘要的保全回执；
  其 ledger 摘要仅绑定封存当时的版本，不限制后续追加。

## 剩余工作

仍明确缺少四类绑定：`complete_execution_sources`、`live_preflight`、
`launch_configuration`、`one_time_scope`。当前只有 package 职责中的
非可执行准备部分，不是完整四模块启动链。

后续按既有设计为证据重建、远端监督和本地主控分别制定可验收实施
切片，并补齐完整启动描述和源码闭包。实时预检及真实 480 项执行
仍须分别核验精确命令、身份和一次性范围，不能用 fixture ID 或
合成记录替代。即使将来动态检查通过，也不等于 LoRA 带来收益；
实际收益仍需另做模型效果对照。

本轮无 SSH/网络、真实 tokenizer/model 加载、生成、训练、安装、
遥测、清理、分支/worktree 切换、commit/push。冻结数据、旧源码、
原八例、评分、4096/256 token 预算和六个已完成新版组件均未修改。
