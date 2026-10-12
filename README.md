# RSI Eval

供 RSI / research agent 调用的三个 benchmark 候选执行与评分接口。
覆盖 MLE Lite 22 题、ALE 40 题、FML 18 题。仓库不实现 LLM agent 或 RSI 搜索策略。

**接入请读 [RSI-agent 接入文档](docs/RSI_AGENT_INTEGRATION.md)。**

```bash
export RSI_EVAL_WORKSPACE=/mnt/afs/yangsen1
./bench list
./bench describe --benchmark ale --task ahc001
```

本仓库仅包含候选评测代码、两个轻量可运行示例、接口测试和文档。
上游仓库、已安装环境、数据、镜像、GPU 配置和运行产物均位于外部准备好的工作区。
当前机器可直接使用；它不是一键从空白机器安装全部 benchmark 的发行包。

- [外部环境契约](docs/ENVIRONMENT.md)
- [验证记录与边界](docs/VERIFICATION.md)
- [上游版本](manifests/upstreams.json)

快速接口测试（不训练、不启动容器）：

```bash
RSI_EVAL_WORKSPACE=/mnt/afs/yangsen1 python3 -m unittest discover -s tests -v
```

真实执行示例：

```bash
./bench validate --benchmark mle --task leaf-classification \
  --candidate examples/mle_leaf_text.py --timeout 180 \
  --run-dir "$RSI_EVAL_WORKSPACE/runs/my-agent/leaf-validation-001"
```

每次调用必须用新的运行目录。公开验证用于搜索；最终隐藏测试成绩只用于评估。
不要把数据、凭据、环境、模型 checkpoint 或全部 runs 提交到 Git。
