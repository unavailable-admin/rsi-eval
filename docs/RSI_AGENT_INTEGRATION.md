# RSI-agent 接入文档

## 1. 本仓库负责什么

本仓库把三个 benchmark 的任务描述、候选执行和评分包装成同一 CLI。
RSI 控制器负责调用模型、生成候选、选择候选、控制预算、保存 agent 版本、调度并发和汇总成绩。
这里没有自动改写 agent 的外层循环，也不调用任何 LLM API。

评估一个 agent 版本时：`describe → 生成候选 → validate → 根据反馈迭代 → 选定候选 → finalize`。
评估多个版本时：对每个版本重复上述过程，控制器再根据内部选择集分数接受或拒绝改写。
最终保留评测集不得参与外层选择。内部选择集的 finalize 分数可以供外层使用，但不得泄露给解题的内层。

## 2. 启动与发现任务

Linux，准备好 Docker / Sysbox 和外部工作区后：

```bash
export RSI_EVAL_WORKSPACE=/mnt/afs/yangsen1
cd /mnt/afs/yangsen1/projects/rsi-eval
./bench list
./bench describe --benchmark mle --task leaf-classification
./bench describe --benchmark ale --task ahc001
./bench describe --benchmark fml --task Fairness_fairlearn
```

工作区通过环境变量指定，仓库可以放在任意目录；数据和环境仍按工作区约定布局。
详细依赖见 [ENVIRONMENT.md](ENVIRONMENT.md)。
`list` 返回 80 个任务：MLE Lite 22、ALE 40、FML 18，尚不覆盖 MLE 全量 75。
`describe` 返回题面文件路径、候选格式、超时范围；FML 另有 `target_files`、`baseline_code_dir`、指标及方向。
控制器应读取题面内容并交给 agent，不是仅把宿主机路径作为提示词。

## 3. 候选格式

### MLE：单文件 Python 适配器

容器执行 `python /agent.py --task /task.json`。脚本应：

1. 读取 `task.json` 的 `competition_id`、`seed`、`public_data_dir`、`output_dir` 和 `budget`。
2. 从只读 `/data` 读取公开训练数据、测试输入和题面；从训练数据自行划分验证集。
3. 训练、验证并预测，写出 `/output/submission.csv`。validate 也要求该文件存在。
4. 写出可选 `/output/candidate-result.json`，报告公开验证成绩，例如：

```json
{"validation_metric":"log_loss","validation_score":0.25,"higher_is_better":false}
```

该 JSON 是适配器自报的验证反馈，统一入口不会独立核实其正确性。没有该文件时 feedback 为 null，
即使进程成功也不表示获得了可比较的验证分数。生产控制器应检查字段、指标方向和验证协议。
不能把 test 输入当成有标签验证集。最终评分器单独读取隐藏标签。

脚本以单文件形式挂载；需要的附属代码必须自行在可写目录生成，或先扩展挂载契约。
容器禁网；LLM 调用、模型或包下载须在控制器/环境准备阶段完成。
宿主机模型缓存并不自动全部挂载到 MLE 容器。
可运行示例：[mle_leaf_text.py](../examples/mle_leaf_text.py)，支持 Leaf 和 Spooky 两题的真实 CPU 训练。

### ALE：C++20 源文件

传入 `.cpp`，按题目标准输入/输出协议求解。使用官方 judge 202301、完整用例集（`lite_version=False`）、4 workers。
validate 只评公开用例；finalize 评公开和私有用例，返回最终排名与 Performance。
可运行弱解法示例：[ale_ahc001.cpp](../examples/ale_ahc001.cpp)。ACCEPTED 仅表示有效执行，不表示高分。

### FML：允许修改文件的目录

先读取 describe 的 `baseline_code_dir`、`target_files` 和任务 prompt。
候选目录按 target_files 中的相对路径存放改动文件。至少提供一个允许的文件；未提供文件沿用基线。
只复制允许修改的文件，其他文件不会被执行器当作补丁应用。必须遵守题面中的修改限制和约束。

```bash
mkdir -p /tmp/fairlearn-candidate
cp "$RSI_EVAL_WORKSPACE/projects/fml-bench/workspace/Fairness_fairlearn/fairlearn/algorithm.py" \
  /tmp/fairlearn-candidate/algorithm.py
# agent 在此改写 algorithm.py，然后执行 validate。
```

每次执行从准备好的官方工作副本复制新 sandbox，再覆盖候选文件。
validate 运行官方验证命令；finalize 在新副本中重新执行验证训练，再执行测试。
因此 finalize 不直接复用之前 validate 的 checkpoint，也需留出训练时间。

## 4. 调用与结果读取

```bash
./bench validate --benchmark mle --task leaf-classification \
  --candidate examples/mle_leaf_text.py --seed 0 --timeout 180 \
  --run-dir "$RSI_EVAL_WORKSPACE/runs/my-agent/leaf-val-001"

./bench finalize --benchmark mle --task leaf-classification \
  --candidate examples/mle_leaf_text.py --seed 0 --timeout 180 \
  --run-dir "$RSI_EVAL_WORKSPACE/runs/my-agent/leaf-final-001"
```

MLE/FML 可加 `--gpu 0`；默认不分配 GPU，GPU 训练任务应显式指定。
ALE 用法相同，候选换成 `.cpp`，不接受 `--gpu` 或 `--timeout`。
FML 候选换成目录，任务如 `Fairness_fairlearn`。

CLI stdout 是一个 JSON 对象，后端日志保存在 runner-stdout.log / runner-stderr.log。
执行结果也写入运行目录的 `suite-result.json`：

```json
{
  "schema_version": 1,
  "benchmark": "mle",
  "task_id": "leaf-classification",
  "action": "validate",
  "state": "succeeded",
  "exit_code": 0,
  "private_grading_requested": false,
  "feedback": {"validation_metric":"log_loss","validation_score":0.25,"higher_is_better":false},
  "backend_report": "/absolute/run/path/report.json"
}
```

`feedback` 保留各 benchmark 原生格式：

| Benchmark | validate | finalize |
|---|---|---|
| MLE | candidate-result.json 内容 | 官方 grading report，含有效提交和奖牌判定 |
| ALE | status、score、cases 等公开汇总 | 私有汇总；顶层另有 rank、performance |
| FML | 各数据集的原生 val metrics | 各数据集的原生 test metrics |

不跨 benchmark 直接平均原始分数。MLE 百分位、FML 归一化改进、跨题均值/标准误由控制器按上游协议汇总。
详细资源看 backend_report：MLE 的 resources/grading_resources，ALE 的 resources/splits，FML 的 splits.val/test。
采样峰值可能漏掉瞬时值；GPU 分配时间不是 GPU 满负荷计算时间。

正常成功退出码 0；候选失败、超时或评分失败返回非零和 state=failed。
参数错误（如目录已存在、候选不存在）在启动前退出，不保证生成 suite-result.json。
不要仅凭文件存在就接受候选，必须检查退出码、state、反馈是否有效及任务约束。

控制器伪代码：

```python
spec = describe(benchmark, task)
for step in range(budget.steps):
    candidate = agent.propose(spec, public_history)
    result = validate(candidate, fresh_run_dir())
    public_history.append(result)  # 成功和失败都记录，按公开指标选择
    if budget.exhausted():
        break
chosen = agent.select(public_history)
final = finalize(chosen, fresh_run_dir())
store_for_evaluator(final)  # 不送回解题 agent
```

## 5. 预算、seed、并发与隔离边界

- MLE/FML 当前执行器固定每容器 8 个逻辑 CPU、32 GiB 内存；默认共享内存 4G，OfficeHome 为 16G。
  它们是当前接口限制，不是论文硬件配置；目前没有 CLI 内存/CPU 调参接口。
- `--timeout` 默认 14400 秒，MLE 分别约束候选执行和最终评分，FML 分别约束 val 和 test。
  **不是**整个 agent 搜索预算；finalize 可能包含多个阶段。控制器必须在自己的外层管理总预算。
- ALE 使用官方逐用例时限。没有整次运行取消/截止时间 API；不要靠仅杀 CLI 父进程管理严格截止，
  这可能留下后端进程或容器。正式大规模调度前应实现进程组取消与容器清理。
- `--seed` 目前只传给 MLE 适配器。ALE/FML 使用原生任务设置；多次 agent seed 由控制器管理，
  不能假设它自动修改 FML 训练随机种子或 ALE 私有用例。
- GPU 使用工作区 runs/gpu-leases 文件锁和显存占用检查，不能协调不使用此锁的其他程序。
  当前 MLE CPU affinity 固定前 8 核；不要直接无约束并发 8 题。FML 按 GPU ID 选择 CPU 核组。
- MLE/FML 候选执行在禁网 Sysbox 容器，ALE 使用官方 Docker judge；控制器在容器外调用 LLM。
  这不是完整对抗性评测安全系统：FML 挂载上游任务与数据资源，仍需遵守任务边界并审计日志，
  不能声称容器已经阻止一切测试数据窥探或 reward hacking。

## 6. 当前验证范围

迁移前已有 80 题准备/基线证据；本仓库迁移后验证目录见 [VERIFICATION.md](VERIFICATION.md)。
验证覆盖接口与执行流程，不代表完成真实 LLM agent 搜索，更不代表复现论文成绩。
PrivacyMeter 曾出现官方约束拒绝，不能把执行成功自动当成有效算法成绩。
