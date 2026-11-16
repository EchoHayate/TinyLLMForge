# 实验阶段发布说明

日期：2026-11-16。

本次发布保存 LoRA necessity 的协议、冻结题库及其修订历史、提示构造、
tokenizer 检查、远程固定函数验证工具、合成测试和阶段结果文档。
代码仍在独立实验目录，不接入主推理引擎。

当前结果以 `20261115-full-matrix-results.md` 和 `README.md` 为准：
1416/1416 项冻结源码×用例已完成远程数据检查；没有模型训练、生成或
LoRA 对照收益。旧日期文档保留当时状态，不应被理解为当前待执行指令。
本次提交不会重跑已完成的验证，也不改变冻结数据或原始封存文件。

## 发布边界

为避免公开内部主机、账号与运行目录，下列原始记录仅保留本地：

- `20261113-existing-host-tokenizer-scope.md`
- `20261113-existing-host-tokenizer-results.md`
- `20261113-existing-runtime-query-scope.md`
- `20261115-dynamic-validation-preflight.md`
- `../../docs/superpowers/specs/2026-11-12-lora-tokenizer-admission-design.md`

`.superpowers/sdd/` 进度记录、报告引用的临时证据目录、原始 SSH 输出、
凭据、模型权重及本次范围外的其他实验也不包含在本次发布中。
因此历史文档中的部分本地证据路径在新克隆中不可用；保留这些引用是为了
说明结果来源，不表示原始运行回执已随仓库发布。

公开的测试可以检查实现与合成回执处理，不能独立复现既有远程运行的全部
原始证据；报告中的摘要也不能代替原始证据。未来实际模型实验仍须独立
完成训练、生成候选隔离执行和完整成本记录。

## 本地回归

使用 Python 3.12，在仓库根目录运行：

```sh
PYTHONDONTWRITEBYTECODE=1 python3.12 -B -m unittest discover \
  -s experiments/lora_necessity -p 'test_*.py' -q
```

测试不连接远程服务器，不启动模型训练或生成，也不重新执行冻结题目。

本次提交前在 macOS / Python 3.12 上执行上述回归：460 项中 458 项通过，
2 项按平台条件跳过，耗时 100.119 秒。另核对既有封存的 250 个私有证据文件
和 51 个工作区文件，字节长度与 SHA256 全部保持一致。
差异检查发现 `tokenizer_output_capture.py` 原有的文件末尾空行提示；
为保留已封存源码的精确字节，本次没有进行无关格式修改。
