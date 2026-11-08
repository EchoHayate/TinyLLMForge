# Feedback adaptation: active verification

先核对反馈，再决定是否修改答案。这里研究的不是“改了多少次”，而是改对了
多少、把原本正确的答案改坏了多少，以及核验是否值得它的成本。

当前实现三层诊断，均不更新模型权重：

- `active_verification_*`：五策略的合成轨迹重放、请求绑定和固定分母计分。
- `development_cpu.py` 与题目/程序模块：四类有限域开发题目、固定候选程序
  的真实 CPU 执行与参考核对。
- `development_protocol.py`、`development_online.py`、`development_model_worker.py`：
  五策略在线模型选择、按需执行本地核验、封存交互后计分。

三层刻意分开：合成计分器拒绝真实材料，固定程序执行器不接受模型生成代码。
历史纯 CPU 结果见 `RESULTS.md`；在线流程见 `online-development-guide.md`。
最新的单连接批次已完成 120/120 条真实模型策略记录，并通过 token、工具
执行与退出/释放核验，见 `ONLINE_MUX_RESULTS.md`。结果没有显示主动核验
收益：所有合法模型最终选择都是 `C`，各策略选择通过率同为 12/24；主动
核验成本更高，另有一次截断回退，不能当作额外选择成功。
此前三次认证失败仍保留在 `ONLINE_RESULTS.md`、
`ONLINE_FOLLOWUP_RESULTS.md` 和 `ONLINE_TRACED_RESULTS.md`。
单连接成功不证明先前认证拒绝的根因已经解决。
这不是 LoRA/SHINE 更新实现，也不把候选选择诊断称为模型自我训练。

## 五种策略

| 策略 | 行为 | 最多模型 / 工具调用 |
| --- | --- | --- |
| `keep` | 保留初始候选 | 0 / 0 |
| `direct` | 直接选择候选 | 1 / 0 |
| `rethink` | 暂定选择后再决定，不增加证据 | 2 / 0 |
| `always_verify` | 先核验，再做一次最终选择 | 1 / 1 |
| `active_verify` | 直接选择，或请求核验后再选 | 2 / 1 |

`always_verify` 不需要先调用模型来请求核验，不能漏掉这个廉价对照。
模型错误、非法输出和截断不重试；缺失记录仍在分母内，未知成本不是零。
选择成功与失败后回退到初始候选的部署结果分开统计，修复与误纠正也分开。
成本比较保留未知项，不用组件耗时冒充完整 episode 耗时。

## 运行

以下本地诊断和测试使用 Python 3.12 标准库，无需安装模型或项目推理依赖。
固定程序执行依赖 POSIX 资源限制，适用于 macOS/Linux；不支持 Windows。
以下命令在仓库根目录执行；真实模型 worker 另需指南所述的已准备环境。

合成诊断只输出 JSON，不执行候选程序：

```sh
python3.12 experiments/feedback_adaptation/demo_active_verification.py --scenario mixed
python3.12 experiments/feedback_adaptation/demo_active_verification.py --scenario ceiling
python3.12 experiments/feedback_adaptation/demo_active_verification.py --scenario incomplete
```

`mixed` 人工安排修复、误纠正和失败；`ceiling` 人工安排各模型策略全部选对；
`incomplete` 缺一条记录并截留另一条轨迹，检查分母和未知成本是否如实保留。
每个场景有 72 个配对格子、360 个预定策略记录；不是独立研究样本。
输出含合成评分答案和语义标签，不能直接作为真实模型提示。

固定程序全域审计和单次核验：

```sh
run_root="$(mktemp -d)"
python3.12 experiments/feedback_adaptation/development_cpu.py audit \
  --output-dir "$run_root/audit"
python3.12 experiments/feedback_adaptation/development_cpu.py probe \
  --task-id dev-002 --initial defect --output-dir "$run_root/probe"
```

输出父目录须已存在，叶目录必须全新；不覆盖、不重试。`audit` 执行 1,209 次
固定函数调用，`probe` 执行指定初始程序的两个固定输入。它们是控制端命令，
不是可直接暴露给模型的工具接口。工具公开投影只取
`result.json` 中的 `comparison.tool_view`，不能传整个结果或私有清单。

每次输出保留源码快照、冻结清单、实际 stdout/stderr、传输记录和比较结果。
这些文件包含本机路径、主机身份及裁决材料；默认留在临时目录，不应直接发布。
`status=completed` 只表示完成解析比较；题目审计是否通过还看
`comparison.admitted`，它也不等于正式 benchmark 准入。

```sh
PYTHONDONTWRITEBYTECODE=1 python3.12 -m unittest discover \
  -s experiments/feedback_adaptation -p 'test_*.py' -q
```

本分支发布的独立文件集合包含 117 项测试，其中部分启动真实固定程序子进程，
模型交互则使用替身。它们不等于真实模型运行证据。
模块使用目录内直接导入；Python API 调用者需将本目录放入 `PYTHONPATH`。

## 答案与执行边界

四题规格和有限输入域公开。参考算法与候选使用不同计算路线，并检查 21 个
显式锚点；正确候选必须全域通过，错误候选必须在开发评分中有语义错误见证。
这些材料均由助手编写，不是独立人工签核，也不能排除共同误解规格。

每题只有两个可见核验输入。两题刻意保留“错误程序通过全部可见核验”的情况：
有限检查通过，不证明程序整体正确。反馈真假仅针对指定输入上的事实主张，
不代表能裁决任意自然语言纠错建议。剩余输入是开发评分材料，不是最终保留集。

worker 使用 `-I -B`、2 秒 CPU 限额、禁用 core dump；控制端等待上限 5 秒。
每个输出流的 2 MiB 上限在捕获后检查，不能防止恶意程序持续输出。
**这不是安全沙箱**：没有内存、文件系统或网络的硬隔离，也未阻止 worker
读取控制端答案。只执行已审阅的固定函数，绝不执行不受信任代码。

`compare(manifest, observation)` 只核对内容与当前源码/主机的一致性，
不是运行来源认证，也不承诺跨机器重放。摘要不是签名，回放不产生新真实运行。

在线诊断固定 24 个配对格子、120 条策略记录，模型只收到公开题面与交互。
最多 144 次模型调用、48 次工具调用；请求和原始输出必须逐条绑定。
部署回退不是选择成功，工具检查通过也不是全域正确性证明。
正式独立评估、净收益门槛和权重更新仍需另行设计与验收。
