---
title: "Wan2.2 部署"
sidebar:
  order: 2
---

本文档介绍如何基于 xLLM 在 Ascend NPU 环境中部署 Wan2.2 视频生成服务。


## 1.拉取镜像环境

首先下载xLLM提供的镜像：

```bash
# A2 x86
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-x86-cann9-20260605
# A2 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-cann9-20260605
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605
```

然后创建对应的容器

```bash
sudo docker run -it --ipc=host -u 0 --privileged --name mydocker --network=host \
 -v /var/queue_schedule:/var/queue_schedule \
 -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
 -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
 -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
 -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
 -v /var/log/npu/slog/:/var/log/npu/slog \
 -v ~/.ssh:/root/.ssh  \
 -v /var/log/npu/profiling/:/var/log/npu/profiling \
 -v /var/log/npu/dump/:/var/log/npu/dump \
 -v /runtime/:/runtime/ -v /etc/hccn.conf:/etc/hccn.conf \
 -v /export/home:/export/home \
 -v /home/:/home/  \
 -w /export/home \
 quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605
```

## 2.拉取源码并编译

下载官方仓库与模块依赖：

```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm 
git submodule update --init --update
```

下载安装依赖:

```bash
pip install --upgrade pre-commit
```

执行编译，在`build/`下生成可执行文件`build/xllm/core/server/xllm`：

```bash
python setup.py build --device npu
```

## 3. 权重准备


模型根目录需要包含 DiT 服务加载所需的组件目录。典型目录结构如下：

```text
Wan2.2-I2V/
├── model_index.json
├── processor/
├── text_encoder/
├── tokenizer/
├── transformer/
├── transformer_2/
└── vae/
```

## 4.启动模型，对外提供 `/v1/video/generation` 接口。

### 杀掉之前的xllm服务化进程

```bash
pkill -9 xllm
```

### 环境变量

```bash
# 0. 加载 Ascend 环境（必须先于 python3 调用）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

# 1. 环境变量设置
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH

# 2. NPU-device 环境变量
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_LEVEL=ERROR
export ASDOPS_LOG_TO_FILE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=7200
export INF_NAN_MODE_ENABLE=0
export INF_NAN_MODE_FORCE_DISABLE=1
export HCCL_IF_BASE_PORT=41433

# 3. 清理旧日志
\rm -rf core.*
\rm -rf log/node_*.log
```

## 启动命令 - 单机拉起样例

```bash
# 4. 推理配置
XLLM_PATH="./build/xllm/core/server/xllm"
MODEL_PATH="/export/home/models/wan2_2"
MASTER_NODE_ADDR="127.0.0.1:17372"
START_PORT=18013
START_DEVICE=8
LOG_DIR="log"
NNODES=8

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"

  ${XLLM_PATH} \
    --model="$MODEL_PATH" \
    --max_memory_utilization=0.98 \
    --backend="dit" \
    --tp_size=1 \
    --cfg_size=2 \
    --sp_size=4 \
    --vae_size=4 \
    --output_shm_size=1024 \
    --devices="npu:$DEVICE" \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --port $PORT \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --use_contiguous_input_buffer=false \
    --enable_rolling_load=true \
    --rolling_load_num_rolling_slots=2 \
    --dit_laser_attention_enabled=true \
    --dit_sparse_attention_enabled=false \
    --dit_sparse_attention_sparsity=0.8 \
    --dit_sparse_attention_sparse_start_step=15 \
    --enable-shm=true \
    --node_rank=$i > $LOG_FILE 2>&1 &
done


日志出现"Brpc Server Started"表示服务成功拉起。




## DiT 参数说明

| 参数 | 说明 | 默认值 | 取值 |
| ---- | ---- | ------ | ---- |
| `--sp_size` | Sequence Parallel 并行度 | `1` | 正整数，如 `1`、`2`、`4`、`8` |
| `--cfg_size` | Classifier-Free Guidance 并行度 | `1` | `1` 或 `2` |
| `--vae_size` | Vae Sparital Parallel 并行度 | `1` | 正整数, 如 `1`、`2`、`4`,  可以与sp_size保持一致|
| `--tp_size` | Tensor Parallel 并行度 | `1` | 正整数, 如 `2`、`4`,  建议不开启，使用动态权重加载|
| `--enable_rolling_load` | 是否启动动态权重加载 | `false` | bool值, true 或者 false|
| `--rolling_load_num_rolling_slot` | 动态权重加载分配槽数 | `2` | 正整数, , 如 `2`、`3`|
| `--dit_laser_attention_enable` | 是否使能laser_attention | `false` | bool值, 如 true，false|
| `--dit_distill_enable` | 是否使能蒸馏模型| `false` | bool值, 如 true，false|
| `--dit_sparse_attention_enabled` | 是否使能稀疏attention| `false` | bool值, 如 true，false；与laser_attention互斥|
| `--dit_sparse_attention_sparsity` | 稀疏attention的稀疏度| `0.5` | float, 如 0.5，0.6；依赖dit_sparse_attention_enabled|
| `--dit_sparse_attention_sparse_start_step` | 开始稀疏的step| `0` | int, 如 5, 10；依赖dit_sparse_attention_enabled|
| `--dit_sparse_attention_version` | 稀疏attention的版本| `rain_fusion` | string, rain_fusion, sparse_attention；依赖dit_sparse_attention_enabled|

`NNODES` 必须等于 `sp_size * cfg_size * tp_size`。

| `sp_size` | `cfg_size` | `vae_size` |`tp_size` |`NNODES` | 说明 |
| --------- | ---------- | -------- | -------- | -------- | ---- |
| `1` | `1` |  `1` | `1` |`1` | 单卡部署 |
| `2` | `1` | `1` | `1` | `2` | 仅开启 SP |
| `1` | `2` | `1` | `1` | `2` | 仅开启 CFG 并行 |
| `2` | `2` |  `2` | `1` |`4` | 同时开启 SP 和 CFG和vae并行 |
 `4` | `2` |  `2` | `1` |`8` | 8 卡部署 |
