# 验证记录

日期：2026-10-12。环境：现有 `/mnt/afs/yangsen1` 准备工作区。
实际运行产物位于外部 `runs/rsi-eval-verification/`，没有把模型、完整日志或数据提交进 Git。

| 检查 | 结果 |
|---|---|
| 全任务目录 | MLE 22、ALE 40、FML 18，共 80 题 |
| describe 文件路径 | 80 题题面文件存在，18 题所有 FML target_files 存在 |
| MLE Leaf validate | 真实 CPU 训练成功，公开 log_loss 0.1406843434 |
| MLE Leaf finalize | 官方有效提交，私有 log_loss 0.12444，与迁移前一致 |
| ALE ahc001 validate | 50 个公开用例 ACCEPTED，score 826577 |
| ALE ahc001 finalize | 1000 个私有用例 ACCEPTED，rank 1232，Performance 383 |
| FML Fairlearn validate | 公开人口统计均等差异 0.1866315885 |
| FML Fairlearn finalize | 测试差异 0.1730295772，accuracy 0.8525884762，与迁移前一致 |
| GPU 透传 | MLE 候选容器 GPU 0 中 CUDA 可用，CUDA 矩阵乘法及同步成功；不是 GPU 训练吞吐测试 |
| 超时 | 休眠候选在 2 秒限制触发超时，非零退出，state=failed，timed_out=true |
| 公开/私有边界 | 三个 validate 报告均无私有评分；MLE 无 grading 目录，ALE/FML 无 private/test 阶段 |
| 仓库可移动 | Git clone 到另一目录后，6 项测试通过，真实 Leaf 训练及公开验证通过 |
| 单元测试 | 6 项通过：后端路径、验证边界、失败封装、旧目录保护等 |

运行目录名：`mle-validate`、`mle-final`、`ale-validate`、`ale-final`、`fml-validate`、
`fml-final`、`fml-final-clean`、`mle-timeout`、`mle-gpu`。
机器可读检查在外部 `checks.json`、`catalog-check.json` 和各目录 suite-result.json / report.json。

这里验证的是迁移后的执行协议，不是重新跑完 80 题训练，也不是 RSI 搜索效果测评。
此前 80 题环境/固定基线证据仍在外部工作区 runs/README.md。
特别是 FML PrivacyMeter 的一次容器候选曾违反官方约束，不应声称所有候选均可得到有效算法成绩。

复核接口测试：

```bash
export RSI_EVAL_WORKSPACE=/mnt/afs/yangsen1
python3 -m unittest discover -s tests -v
```

真实执行命令见接入文档；重跑必须选新目录，保留历史记录。
