# 外部环境契约

本仓库使用已经准备好的 Linux 工作区，通过 `RSI_EVAL_WORKSPACE` 指定。代码路径与该路径独立。
当前可用实例为 `/mnt/afs/yangsen1`。移植到其他机器需要重新准备依赖；复制虚拟环境并不保证可用。

```text
$RSI_EVAL_WORKSPACE/
  projects/{mle-bench,ale-bench,fml-bench}/
  envs/mlebench/                 # 主控制器，docker、psutil、mlebench、pandas 等
  envs/mle-gpu/                  # MLE 候选运行 Python、PyTorch、sklearn 等
  envs/alebench/                 # 官方 ALE 及 Docker SDK
  envs/fml/<conda_env>/          # FML 各任务环境
  tools/python/                 # MLE 环境依赖的 Python 解释器
  data/mlebench/<id>/prepared/{public,private}/
  data/alebench/                 # 官方 zip、problem_ids.txt、problem_ids_lite.txt
  data/alebench/prepared/<id>/tools/target/release/  # 已编译官方 gen/vis/tester
  data/fmlbench/
  cache/{huggingface,torch}/
  runs/prepare-benchmarks/ale/statements/<id>.md
  runs/configs/sysbox-gpu<N>.json # 本机设备/驱动挂载配置，不含凭据
  runs/gpu-leases/               # 与已有调度器共用的 GPU 锁
```

FML `projects/fml-bench/workspace` 也必须已准备好；其配置 repo_dir、任务数据链接与环境一致。
上游 commit 固定值见 [upstreams.json](../manifests/upstreams.json)。
这些 commit 固定源码，不自动固定准备后的数据、下载模型和所有 Python 包；精确环境证据保存在
外部 `runs/prepare-benchmarks/environment-snapshots/` 与 `runs/baseline-resources/frozen-baselines.json`。

本机镜像：MLE/多数 FML 用 python:3.11-slim，旧 PyTorch FML 环境用 ubuntu:22.04；
ALE 用对应 C++20 judge 202301 镜像。所有镜像必须提前准备。
Docker daemon 需可访问，MLE/FML 使用 sysbox-runc，GPU 使用显式设备和只读驱动库挂载。
现有 Sysbox 配置不能直接替换为 Docker `--gpus`；设备编号或驱动变更后需重新生成并验证配置。
不要把主机 Docker 配置、API key、Kaggle 凭据提交到仓库或传入候选容器。

本仓库保留 MLE 评分时的 Pandas 字符串兼容选项；不修改官方评分代码。
原工作区的数据下载、环境安装、固定弱基线大批量运行脚本未复制进来，避免混淆接入所需代码。
