# LoRA 题库：远程 D01 动态核验

2026-11-15。用户确认在远程执行并要求继续接好执行、先跑首题、再核验完整题库。
本次实施止于首题 D01 的 32 项实际调用，不训练、不生成、不执行其他题目。
本设计替代已停用的本地 sandbox-exec 方案。

## 已取得的启动证据

既有 SSH 前缀可登录 n232-195-203，UID 1054，Python 3.12.13 可启动。
远端 bwrap 0.3.1 是 setuid 安装。本次人工探针先设置
PR_SET_NO_NEW_PRIVS，再运行 bwrap --unshare-all，固定 /usr/bin/true 返回 0。
不依赖提权，不改变安装，不使用共享 Docker 管理接口。
这只是启动证明，不是安全版本背书或题库执行结果。

首次完整人工探针在构造seccomp时报告不认识clone3，尚未启动任何候选。
兼容处理只在libseccomp报告原生AUDIT_ARCH_X86_64时将未知clone3映射到435；
其他未知名称/架构仍拒绝。Linux v6.12官方
`arch/x86/entry/syscalls/syscall_64.tbl`核对了该编号，
原文件SHA256为`980ce3115028c71c5618e7864d262017bde8103bcfe7b413147a14fd312c92ac`。
禁止规则不被省略，不更新系统库。后续人工探针必须直接验证该编号被拒绝。

## 实施选择

采用现有 bwrap 的非提权 user/mount/pid/network/ipc/uts namespaces，
加 libseccomp 拒绝 socket、fork/vfork/clone/clone3、unshare/setns、
ptrace/process_vm_readv/process_vm_writev。不用普通宿主执行回退；
也不为本批创建 Docker 镜像、安装库或修改服务。库或能力缺失即报告实际错误。

只挂载只读系统运行库和实际 Python 基础运行时（重定位为 /runtime），
不挂载 home、模型、仓库、凭据、控制端材料、宿主 /proc 或 Docker socket。
容器有自己的 proc、dev、临时目录，父端只传 PATH/HOME。
bwrap会加入PWD=/work，Python可能加入LC_CTYPE=C.UTF-8；
探针核对这些具体值，不能因运行库补充局部元数据而误判凭据继承。
PR_SET_NO_NEW_PRIVS 在远程受信控制程序内设置，在启动 bwrap 前生效。
候选继承 seccomp；重执行不能解除其限制，不宣称 exec 本身被禁用。
工作包装器设置 CPU 2 秒硬限、CORE=0、FSIZE=0、NOFILE=32，
不恢复用户取消的聚合内存/磁盘预算，也不要求远端自动清理。

威胁范围是当前冻结、已静态审查的小函数，不准入任意模型生成的程序，
不声称内核或旧 bwrap 已完成全面安全审计。
固定人工探针须先验证命名空间不同、宿主路径不可见、运行时只读、
网络/fork 被拒绝及超时/输出上限生效。

## 边界和顺序

冻结数据及 validation_contract.PINS 不变。完整清单 1416 项，首批前32项
全部属于 D01。只上传 worker_requests 的 request_id/source/entry_point/args，
不上传 expected、witness_prediction、评分理由或整个任务对象。
每项新进程，一次调用，无重试；先运行人工探针，再审查代码和探针，
再以新 run_id 执行一次 D01。失败记录保留，不复用消耗的运行目录。

单项墙钟2秒，全批90秒，stdout65536字节、stderr4096字节；
控制端非阻塞采集，遇超时或超限杀掉自建进程组并收取退出状态。
不正常执行停止剩余批次，缺项保留。外层SSH收集上限8MiB/128KiB，
110秒（包含全批期限），即使断线也不把远端未知状态写成清理成功。
任务正常返回与预期不符是语义结果，不当作基础设施失败。

## 接口

- validation_task_worker.py：stdin单请求，stdout单个return/raises ValueError；
  没有答案，非允许异常返回非零且无栈。候选额外输出不能被丢弃后冒充合法结果。
- validation_remote.py：collect_process、bwrap配置/seccomp、run_bundle、JSONL输出。
  不导入仓库控制端，不接触正确答案。可作为完整文本通过远程python -c加载。
- validation_real_report.py：build_d01_report(source_raw, manifest_raw, bundle_raw,
  transport)。重建清单并严格绑定实际观察；不调用simulate_validation。

bundle字段恰为 kind, mode, run_id, worker_source, requests。
kind="LORA_REMOTE_VALIDATION_BUNDLE"，mode="probe"或"d01"，run_id为32位小写hex。
只有d01可进入build_d01_report；requests须逐字节等价于完整投影前32项。
worker_source是实际执行包装器文本，不是候选源码。

collect_process返回恰为 status, returncode, stdout_hex, stderr_hex, elapsed_ns,
process_reaped, stdin_complete。status为completed/program_error/timeout/
output_limit/launch_failed。二进制输出以hex无损保存，不能先忽略编码错误。
任务层completed需returncode=0、空stderr、stdin_complete及process_reaped均true。
外层SSH沿用带-vvv的成功前缀，stderr保存认证诊断，不要求为空；
只在外层采集时显式设置stderr_is_diagnostic=True，不丢弃或截去其原始字节。
其余结果不作功能比较。返回码为int或null，时间为非负精确int。

远端JSONL事件：

- ready: kind, run_id, bundle_sha256, worker_sha256, runtime。
- row: kind, run_id, ordinal, request_sha256, result。
- done: kind, run_id, received, elapsed_ns, stopped。

runtime是受信远端程序记录的字典，包含host/uid、Python与bwrap二进制摘要、
seccomp摘要、实际argv前缀及no_new_privs；不是候选自报的隔离证明。
本地报告保存runtime，最终启动driver核对probe与d01的runtime完全相同。
本地执行driver另保存完整命令与backend源码摘要。
driver将已通过probe的runtime作为EXPECTED_RUNTIME字面量绑定到远程启动代码；
run_bundle在ready及首项调用之前核对，d01缺少或不符即失败。
这不是等题目执行完才检查，也不把期望答案加进远端bundle。
ready需首条且唯一，row须ordinal从0严格递增至31，done末条且唯一；
中途停止/缺done保持不完整，拒绝重复、乱序、错误ID/摘要和额外字段。
所有输入按现有严格JSON规则拒绝重复键、非有限值、未知输出字段。
transport须包含上述collect_process字段，其stdout就是原始JSONL。

报告 kind="LORA_D01_EXECUTION_REPORT"，full_scheduled=1416，
batch_scheduled=32，received实数，remaining_full=1416-received。
32行均保留，缺行status=missing，expected_match/prediction_match为null。
reference_ok检查8个reference结果；initial_scoring_rejected至少一个评分拒绝；
witnesses_ok检查恰好3个冻结见证：预测符合且正确期望不符合。
batch_complete需全部32项正常、完整done及外层成功；
batch_checks_satisfied需batch_complete且三种判断都true。
始终full_dynamic_validation_complete/benchmark_eligible/formal_evaluation_ready/
net_benefit_established/isolated/generated_candidate_backend_ready=false。
正常批次是实测观察，测试夹具不是实际运行证明；
报告验证器仍以受信控制端和远端启动链路为信任前提，不是密码学执行证明。

## 不变约束

继续使用现有实验分支，不建分支/worktree、不提交/push、不安装或发遥测。
标准库unittest，先RED再GREEN；旧模拟API/数据/tokenizer不改。
只修改上述三个模块、对应测试及本阶段文档；保留其他未跟踪文件。
证据在新的0700目录，文件0600。全部完成后才更新README的真实调用数。
