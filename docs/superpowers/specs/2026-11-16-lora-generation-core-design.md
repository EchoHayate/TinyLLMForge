# 固定参数候选生成组件

日期：2026-11-16。承接已批准的 LoRA necessity 四组对照设计，以及真实
16 步训练链路完成后的继续指令。本轮落实候选生成这一前置组件，不重训，
不运行迁移/保留评分或生成代码，不把组件完成当作远程评测准入。

## 实现选择

1. 采用小型、无 KV cache 的逐 token greedy 循环：直接复用既有模型 forward，
   不继承模型 generation_config 中可能影响采样/惩罚/停止的隐含配置。
   代价是重复计算前缀；这是正确性参考组件，不是生产吞吐方案。
2. 直接包装 Transformers generate：可利用缓存，但需另行验收完整解码配置、
   logits processor 和 cache 所有权。本轮不新增这层依赖。
3. 直接接主推理引擎：扩大状态管理及适配器热切换范围，不符合独立实验边界。

选择第一项；不改变原协议的 greedy、4096 输入及 256 输出 token 上限，
也不新增采样、重试或 checkpoint 搜索。

## 接口与责任

`generate_candidate(model, tokenizer, messages, *, expected_tokenizer_sha256) -> dict`

输入是已由 `prompt_rendering` 构造的单轮或第二轮消息。调用方负责组别/任务
身份、同信息配对、只读 adapter 挂载、底座/adapter 完整性、运行级时限、
消息与冻结任务绑定及错误回执。组件不读题库、评分答案、权重文件或网络。
组件不声明输入来自哪个 arm，也不自行创建监督内容。

组件只接受单设备 CPU/CUDA、eval、无梯度的 torch Module。CPU 仅用于合成
验证；禁止 autocast。torch 使用函数内导入，保持现有 discovery 冷启动测试。
组件不调用 `.train()`、`.eval()`、`.requires_grad_()`、优化器或 backward；
不自行安装、挂载或卸载 adapter。外层运行器须独占模型；此函数不是并发服务。

## 算法与验证

1. 验证预期 tokenizer SHA256 格式；深拷贝消息，使用既有
   `tokenizer_measurement.measure(..., answer=None)` 构造 prompt。要求状态
   摘要匹配、status OK、输入不超过 4096，无截断、无 thinking。
2. 检查模型/参数/缓冲区同设备、eval、参数不需要梯度且 grad 为 None。
   普通 BF16/FP32 浮点状态及整数缓冲区可接受；拒绝无状态模型、
   inference tensors 和非 dense 状态。不要为补兼容性改变调用方状态。
3. 使用 `torch.no_grad()`，每步把完整 prefix（prompt 加此前输出）及全一
   attention mask 传给模型，明确 `use_cache=False, return_dict=True`，
   不传 labels、past_key_values 或其他会携带前题状态的参数。
4. 要求返回 logits 是浮点张量，形状为 `(1, 当前序列长度, vocab)`；
   仅末位置 logits 参与本次选择，要求其全部有限。取 argmax，平分时按
   torch argmax 的首位置规则；所选 ID 必须属于 tokenizer 词表。
   不能把非法最大值屏蔽后另选 token。
5. 第一个 EOS 立即停止并计入 output_tokens；否则恰好 256 个新 token 停止。
   stop_reason 分别为 `eos` / `length`；length 不代表候选有效或通过。
   空候选允许返回，由后续解析器判 invalid，不能伪造成生成异常。
6. 解码只去掉终止的 EOS，使用 `skip_special_tokens=False`、
   `clean_up_tokenization_spaces=False`，不 strip、不去 Markdown、不修复源码。
   必须返回可编码 UTF-8 的字符串，最多 65536 bytes。重新验证 tokenizer
   状态未漂移及消息未修改。只检查 tokenizer，不再次构造/改变提示。
7. 返回输出原文、完整输出 token IDs（含 EOS）、其摘要和原始输入摘要、
   token 数、forward 次数、停止原因、tokenizer 摘要和费用。一个调用
   `model_calls=1`，`forward_calls=output_tokens`，两者不可混淆。

接口输出：

- `kind="LORA_GREEDY_CANDIDATE_UNATTESTED"`、`device_type`；
- `tokenizer_state_sha256`、`messages_sha256`、`prompt_sha256`；
- `input_tokens`、`output_tokens`、`output_ids`、`candidate`、`candidate_sha256`；
- `stop_reason`、`model_calls`、`forward_calls`；
- `timings_ns`：`preparation`、`generation`、`decoding`、`wall`。
  前三项为互不重叠阶段，wall 包含它们与其余校验；GPU 阶段边界同步。
- `peak_allocated_bytes`、`peak_reserved_bytes`：CUDA 测量整个调用并注明包括
  驻留模型；CPU 为 None。只在输入/model 检查通过后 reset CUDA peaks。
- `benchmark_eligible`、`formal_evaluation_ready`、`net_benefit_established`、
  `isolated` 全部 false。

错误直接抛出，不改候选、不重试、不返回成功回执。已经发生的模型调用
无法撤销；外层控制器必须保存失败与已消费调用，不得把异常当作未执行。
no_grad/eval 不是参数完整性证明；底座/adapter 字节校验由外层负责。
无 cache 参数不证明任意自定义模型没有内部隐藏状态，准入仅针对经审阅模型。

## 本轮验收

脚本化小张量模型检查 prompt 前缀增长、greedy 与 EOS/length、输入不超限、
词表/模板绑定、三角色第二轮、no_grad/no_cache、非法 logits、错误传播、
输出原样保存和 JSON-native 成本。测试不在 discovery 时 import torch。
另用既有依赖构造随机 tiny-Qwen，执行真实 forward API 的 CPU 兼容性检查，
不加载预训练 checkpoint、不读取评分答案；独立审查后记录结果。

保留既有源码、数据、训练产物、失败记录及封存原件；只新增组件/测试/本轮
文档。保持当前实验分支，不创建 worktree、不提交/push、不安装、不遥测、
不更新记忆。测试技能的 bootstrap/telemetry 与此边界冲突，沿用既有 unittest。
本阶段不连接远程。下一步远端只可先做 4 道开发题 base/text 的协议检查；
该阶段仍需生成器、隔离执行器与两轮回执整体验收，不能直接跳到 32 题评分。
