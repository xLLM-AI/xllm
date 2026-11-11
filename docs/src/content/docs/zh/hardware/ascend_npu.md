---
title: "昇腾 NPU"
description: "使用 NPU 后端在昇腾硬件上运行 xLLM。"
sidebar:
  order: 2
---

本页介绍如何配置 昇腾 NPU 环境、安装 xLLM，并按 [启动 xllm](/zh/getting_started/launch_xllm/) 使用统一的 `xllm serve` 命令启动服务。

## 前置准备

- 在 Linux 宿主机上安装昇腾驱动和 Docker。确认 `npu-smi info` 能识别 NPU，并选择与 A2/A3 硬件、宿主机 CPU 架构匹配的镜像。
- 在 [模型支持列表](/zh/supported_models/) 中确认后端支持目标模型，并将权重放在容器内可访问的目录。
- 下方命令将宿主机主目录挂载到容器内同一路径。若权重存放在其他位置，请增加类似 `-v /path/to/models:/models:ro` 的挂载，并在 `MODEL_PATH` 中使用容器内路径。

## 启动容器

以下命令在宿主机执行。根据硬件选择一个开发镜像：

| 硬件 | 宿主机 CPU | 开发镜像 |
| --- | --- | --- |
| Ascend A2 | x86_64 | `quay.io/jd_xllm/xllm-ai:xllm-dev-a2-x86-cann9-20260911` |
| Ascend A2 | ARM64 | `quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-cann9-20260911` |
| Ascend A3 | ARM64 | `quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260911` |

下方示例选择 A3 ARM64；其他平台请修改 `XLLM_IMAGE`。

```bash
XLLM_IMAGE="quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260911"
docker pull "$XLLM_IMAGE"
```

```bash
docker run -it \
  --name xllm-npu \
  --ipc=host \
  -u 0 \
  --privileged \
  --network=host \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /usr/local/sbin/:/usr/local/sbin/ \
  -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
  -v /var/log/npu/slog/:/var/log/npu/slog \
  -v /var/log/npu/profiling/:/var/log/npu/profiling \
  -v /var/log/npu/dump/:/var/log/npu/dump \
  -v "$HOME:$HOME" \
  -w "$HOME" \
  "$XLLM_IMAGE" \
  /bin/bash
```

使用更多设备时，请在创建容器前增加对应的 `--device=/dev/davinciN` 参数。

退出后容器会保留，使用 `docker start -ai xllm-npu` 重新进入已停止的容器；若容器仍在运行，使用 `docker exec -it xllm-npu bash`。

## 安装 xLLM

以下命令在容器内执行。若镜像已安装 xLLM wheel 且支持 `xllm serve`，可跳过源码安装。否则，克隆仓库（或进入已有源码目录）、初始化子模块，并为当前后端构建和安装 wheel：

```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm

# 第一次需要进行 pre-commit 安装
pip install pre-commit
pre-commit install

git submodule update --init --recursive
python setup.py install --device npu
```

也可以直接使用 `python setup.py install` 安装；省略 `--device` 时，安装脚本会根据当前 PyTorch 环境自动识别设备后端。

NPU wheel 会携带构建环境中完整的 xllm_ops vendor，包含动态库、算子描述、tiling 库和 kernel。打包时与构建使用相同的选择规则：如果 `${ASCEND_OPP_PATH}/vendors/custom_xllm_math/op_api/include/aclnnop` 目录存在，则选择 `custom_xllm_math`；否则选择旧版 `${ASCEND_OPP_PATH}/vendors/xllm`。两种布局均打包到启动器和原生库使用的固定路径 `xllm/opp/vendors/custom_xllm_math`。打包时要求已安装的算子资源完整，且其版本与 `third_party/xllm_ops` 的 Git 提交一致；资源缺失或版本不匹配会使构建失败。其他 vendor 不会一并打包，模型所需的额外算子包仍需按对应部署说明准备。

已有 wheel 时，可以直接在目标机器安装，无需重新编译。将下方路径替换为实际 wheel 文件：

```bash
python -m pip install /path/to/xllm_npu*.whl
```

目标环境仍需具备与 wheel 兼容的 CANN、驱动、硬件以及 Python/PyTorch 环境，但无需另行安装 wheel 已包含的 `custom_xllm_math`。`xllm` 启动器会自动配置该算子包的动态库和自定义 OPP 搜索路径，保留基础 CANN OPP 配置。

`python setup.py build` 仅构建产物；统一启动命令需要安装 wheel 及其包含的服务二进制。继续前先检查已安装的命令：

```bash
command -v xllm
xllm serve --help
```

## 启动服务

将 `MODEL_PATH` 替换为权重目录后，在容器内执行。下方示例启动一张卡；请结合目标模型部署配方和 [CLI 参数参考](/zh/cli_reference/) 调整。

```bash
#!/bin/bash
set -e

export ASCEND_RT_VISIBLE_DEVICES=0

MODEL_PATH="/path/to/model/Qwen3-8B"
PORT=8010
MASTER_NODE_ADDR="127.0.0.1:9748"
NNODES=1

xllm serve \
  --model="$MODEL_PATH" \
  --model_id=Qwen3-8B \
  --port="$PORT" \
  --nnodes="$NNODES" \
  --master_node_addr="$MASTER_NODE_ADDR" \
  --block_size=128 \
  --communication_backend=hccl \
  --max_memory_utilization=0.86 \
  --enable_prefix_cache=false \
  --enable_chunked_prefill=true \
  --enable_schedule_overlap=true \
  --enable_shm=true \
  --enable_graph=true
```

单机双卡时，将脚本中的设备选择和 `NNODES` 两行替换为：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
NNODES=2
```

只需执行一次 `xllm serve`。启动器自动分配 rank 和各 rank 的端口，将日志写入 `log/node_0.log`、`log/node_1.log` 等文件，并在前台运行。不要传入 `--node_rank` 或再套一层按设备启动的循环。

## 验证服务

在另一个 shell 中，等待模型加载完成后查询 rank 0 的服务（本例端口为 `8010`）：

```bash
curl http://127.0.0.1:8010/health
curl http://127.0.0.1:8010/v1/models
```

模型列表应包含 `--model_id` 指定的 `Qwen3-8B`。调用 [OpenAI 兼容 API](/zh/basic_usage/openai_api/) 时使用该 ID；若修改 `PORT`，请求地址也要同步调整。

## 平台说明

- 示例使用 HCCL、`--block_size=128`、共享内存、Chunked Prefill、调度重叠和图执行。修改这些设置前，请先查看对应模型的部署配方。
- 同时运行多个独立服务时，分别配置 Master 地址、服务端口范围、日志目录和 HCCL 基础端口。
- 多机 NPU 部署还可能需要 `--rank_tablefile`。网络和 rank table 准备见 [多机部署](/zh/basic_usage/multi_machine/)，启动命令使用上文的统一启动器。
- Kernel 开发见 [昇腾 NPU TileLang Kernel 开发](/zh/dev_guide/tilelang_ascend_kernel_dev/)。
