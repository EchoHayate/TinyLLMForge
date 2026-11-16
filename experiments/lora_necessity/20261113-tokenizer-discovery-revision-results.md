# Tokenizer discovery revision — November 13, 2026

## 结论

**依赖查询的误拦截已修复并通过本地测试；新的真实远端检查仍未完成 tokenizer 初始化。**
服务器连接和结果回传正常。本轮在 `TOKENIZER_LOAD` 阶段出现 `ImportError`，
同时 `guard_failed=true`。324 行均未开始计数，不能判断 token 长度是否满足限制，
也没有获得 LoRA 效果或 RSI 的证据。

这次能确认失败发生在 tokenizer 加载阶段并命中了禁止模块拦截；**不能确认具体
模块、导入链或唯一根因**。不能仅凭这个标记认定已经加载模型权重，也不能把所有
限制直接放开。此前发现的依赖查询缺陷不是已经证明的远端唯一原因。

## 本轮改动

- 保留实际 import 的首位拦截器；给公开的 `importlib.util.find_spec` 增加受控查询
  包装。返回真实的存在/不存在结果，不执行禁止模块；返回的 spec 也不能用于执行
  原加载器。需要导入禁止父包的嵌套查询仍被拒绝。
- 检查查询包装器的所有权和完整性；关闭时只恢复自己安装的包装器。
- 初始化失败只增加固定阶段、固定异常类别和布尔拦截状态，不携带异常原文、
  自定义类型名、提示词、答案、token ID 或 traceback。
- 审查发现旧式 finder 的兼容性问题，已按 Python 3.12 行为修复：
  跳过缺少 `find_spec` 方法的 finder，但不吞掉方法体内抛出的异常。
- 没有修改冻结输入、五个 tokenizer 文件 pin、计数顺序、4096/256 token 限制、
  遇首个错误停止的规则或科学验收标记。

## 验证

- 修复前保留 RED 证据；最终全量 LoRA-necessity 测试：
  **362 项，360 通过，2 项 Linux 专用测试跳过，28.188 秒**。
- 官方 Transformers 4.57.1 中已检查的查询 helper，在本地真实临时文件/分发元数据
  fixture 上通过“存在”和“不存在”两种检查，未导入 safetensors 或 Transformers。
  这不是完整远端库的兼容性证明。
- 两组独立审查及最终跨组/精确启动配置复审通过，零遗留发现；初轮 P1 记录保留。
- 新报告通过实际 `build_report` 精确重建。旧轮真实报告也能在修订后接口下精确
  重建，不改写旧证据。

## 新的真实远端结果

- Run ID：`84533a4ce05649df8f4e94f826dbbbd6`。
- Manifest SHA256：
  `3a9d1edf531a2b363871a5d419757951fbb557cf79ae27c5ddecded423c2cdc9`。
- 使用既定 SSH 跳板、服务器和现有 Python/tokenizer；只发生一次新远端连接。
- READY 的主机、用户及禁止创建子进程/线程的实际检查通过后才发送私密输入。
- SSH 退出码 0；约 5.04 秒；上传 2,240,873 字节；
  完整协议回传 1,688,519 字节，无结果截断。
- `OPEN_FAILED`；`TOKENIZER_LOAD / ImportError / guard_failed=true`；
  会话未终结。未完成测量，因此零溢出计数不代表不存在溢出。
- 应用 stdout 为零字节；stderr 的 159 字节精确匹配标准“模型后端不可用、
  tokenizer/configuration 工具仍可用”提示，不是具体失败模块或异常栈。
- 首次本地调用在 SSH 创建前因 `/var` 非规范路径被拒绝。保留该失败及初始 claim；
  改用同一目录的 `/private/var` 真实路径后发起了唯一远端连接。
  两次本地 `execute_once` 调用不等于两次远端运行。

| 类别 | 计划行数 | 已计数 | 未计数 |
| --- | ---: | ---: | ---: |
| 训练样例 | 8 | 0 | 8 |
| 首轮提示 | 136 | 0 | 136 |
| 参考答案 | 44 | 0 | 44 |
| 样例第二轮 | 136 | 0 | 136 |
| 合计 | 324 | 0 | 324 |

没有远端重试、模型生成、LoRA 训练、GPU 作业、依赖安装、清理、提交或推送。
沿用旧报告的 `CLEANUP_UNVERIFIED` 和 false 科学标记；远端残留状态未另行检查。
本轮未把自动清理或已取消的聚合资源预算重新设为前提。

## 下一步边界

先定位触发拦截的固定模块类别及其导入链，区分 tokenizer/configuration 所需
工具模块与真正的模型执行依赖，再决定是否需要新的最小修订。
不能通过笼统解除 backend/modeling 限制来取得“通过”。
本轮唯一远端连接已使用，后续检查必须形成新的明确范围和运行记录。

## 证据

私密根目录：
`/private/var/folders/0c/hl2r_50x28j2ysjvv2120t8h0000gn/T/tlf-tokenizer-discovery-20261113-pdcqnt4i`。

- `authorization.json`、`authorization-run-v2.json`、`connection-claim.json`。
- `local-prelaunch-rejection.json`、`attempt-claim.json`：保留本地启动前拒绝记录。
- `candidate-v2/`：实际运行的冻结源码、profile、输入和 manifest；旧 candidate 未覆盖。
- `review/`、`review-v2/`：初轮发现、修复复审、跨组审查与 `review-v2/report.html`。
- `all-lora-tests-v2.log`、`local-discovery-fixed.json`：本地验证。
- `attempt/`、`validated-result.json`：私密原始协议和完整重建报告，禁止公开。
- `public-outcome-summary.json`、`diagnostic-classification.json`：结果摘要和提示分类。
- `old-report-compatibility.json`、`verification.json`：旧报告兼容性与最终证据校验。
