# 真实底座：非任务生成验证

日期：2026-11-17。接续用户授权的连续实验工程实现，不修改冻结评测。

## 目标和排除项

随机小模型的既有版本兼容性已经通过，但它不证明真实 tokenizer、
Qwen3-8B buffers、CUDA 分支及两轮消息形状兼容。
下一关仅检验这些接口，不训练、不挂载 adapter、不打开正式题目，
不执行或评分模型输出。不重复之前的训练或已消费探针。

采用保留真实训练清单里的同一底座、tokenizer、解释器、75 项版本及 GPU
身份；清单只是已有来源，启动时仍重新检查文件和设备。
仅 base 模型，这与后续开发阶段 base/text 两组一致。

## 固定工作量

两个预声明的合成消息序列，各一次 `generate_observed`，保持原生成上限：

1. `[{"role":"user","content":"For an API compatibility check, reply briefly."}]`
2. `[{"role":"user","content":"For an API compatibility check, reply briefly."},
   {"role":"assistant","content":"Ready."},
   {"role":"user","content":"Now reply briefly again."}]`

第二组的 assistant 文本预先固定，不是第一组输出或真实任务答案。
每次正常调用前额外保留一次 tokenizer 测量（单列耗时），用其原始 prompt、
token IDs 和摘要核对实际生成记录；不增加模型生成或参考生成。
没有 reference 生成，没有正确性判分，也没有失败后的替代输入。
总计两次正常生成请求，每次最多 256 次 forward；EOS 可以提前结束。

另做 tokenizer-only 合成输入预算检查。用重复的 `" x"` 构造经实际
tokenizer 计数恰为 4096 与 4097 的 user 消息；每个目标最多 16 次测量，
找不到精确边界则中止，不假装测试通过。
4096 只测 tokenizer，不声称模型具备最大输入容量。
4097 通过一次独立、预声明的观察封装调用验证输入超限拒绝，
要求两个 forward 计数都是 0，不产生 generation 记录。
同时要求确切的 input-limit 原因、匹配的 claim/消息/tokenizer 身份，
且没有清理、持久化或证据收集错误；其他零 forward 故障不能算预期拒绝。
此项是预算拒绝诊断，不计为实际模型生成；其尝试/成本仍单独记录。

## 源码、状态和资源

采用独立私有目录和独占 claim，保留确切源码、输入及原始输出摘要。
`-I -S -B`，显式已知包路径、offline/no-telemetry 设置；不运行远程代码下载。
加载前重新核对所有 checkpoint/config/tokenizer 文件的 SHA256，
加载后核对 size/mtime/inode 与加载前一致。
BF16、eager attention、eval、frozen、禁用 cache/autocast/TF32，
单张已知 A100；启动前检查该设备没有其他 compute process。
复用此前启动检查的 idle 条件：free 至少 65536 MiB、used 不超过 512 MiB；
这是避免干扰其他 GPU 工作的准入检查，不是新增的实验总内存配额。
设备可用性不等于永久独占租约；不终止其他人的进程或更改服务。

复用既有 `_state`/`_tensor_digest` 的精确规则，比对保留底座摘要，
运行前后 tokenizer 身份及底座状态不变、无梯度。只调用哈希工具，
不调用训练模块里的任何训练或挂载入口。

内部截止时间 540 秒；独立远端父进程 600 秒后 TERM、再 2 秒 KILL，
本地连接最多 620 秒。这里的逐次诊断截止时间不恢复用户已取消的
总内存/磁盘预算或断连自动清理要求。

## 失败和验收

采用已独立验收的观察封装保存每次 claim、完整结果或 failed observation。
打印源/运行时验证、设备观察、加载完成、每次调用开始及完成、最终 DONE
等阶段记录。错误保存可观察计数、墙钟和异常信息，退出非零，不重试。
原始设备记录、tokenizer 状态与边界测量、正常调用前测量和拒绝记录都导出至
stdout；异常时尽力导出当前调用已有记录及收集错误，不用收集错误覆盖原始失败。
硬杀或断连没有末尾 DONE 时一律不完整；未知计数不能写零。

通过要求：

- 源码、模型、tokenizer、依赖清单匹配，两个正常调用均返回完整记录；
- 两个正常调用的 root forward 前后计数都等于对应输出 token 数；
- token/字节/停止规则正确，CUDA allocated/reserved 峰值为实际正整数；
- 4097 拒绝样例两个 forward 数为 0，4096 仅有 tokenizer 证据；
- 原始输出、返回码 0、身份及末尾 DONE 共同满足；参数与 tokenizer 不变；
- 独立重建记录与成本边界，不把嵌套时间段相加。

正常加载、摘要检查、tokenizer 边界探测、两次生成、预算拒绝、
连接与验证成本分别保留。通过只表示真实模型非任务生成链路可用，
不表示纠错有效、32 题可运行或 LoRA 比文本记忆更值得。
