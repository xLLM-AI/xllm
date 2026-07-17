---
title: "MiniMax-M2.7"
sidebar:
  order: 3
---

+ 源码地址：https://github.com/xLLM-AI/xllm

+ 国内可用: https://gitcode.com/xLLM-AI/xllm

+ 权重下载: [modelscope-MiniMax-M2.7](https://www.modelscope.cn/models/MiniMax/MiniMax-M2.7)
+ 离线反量化权重: [modelscope-Minimax2.7-BF16-xLLM](https://modelscope.cn/models/Eco-Tech/Minimax2.7-BF16-xLLM)

## 0.权重准备

MiniMax-M2.7 原始权重为 FP8 格式，xLLM 支持以下三种方式加载：

### 方式一：直接加载 FP8 权重（在线反量化）

直接使用原始 FP8 权重路径，xLLM 会在推理时在线将 FP8 反量化为 BF16 计算，无需额外处理。

```bash
MODEL_PATH=/path/to/MiniMax-M2.7/
```

### 方式二：离线反量化

使用工具脚本预先将 FP8 权重转换为 BF16 格式，避免在线反量化的额外开销：

```bash
python tools/dequant_minimax_fp8.py --input-dir /path/to/MiniMax-M2.7/ --output-dir /path/to/MiniMax-M2.7-bf16/
```

### 方式三：下载预转换的 BF16 权重

直接下载已反量化好的 BF16 权重：

```bash
git clone https://www.modelscope.cn/Eco-Tech/Minimax2.7-BF16-xLLM.git
```

## 1.拉取镜像环境

首先下载xLLM提供的镜像：

```bash
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-20260429
```

然后创建对应的容器

```bash
sudo docker run -it --ipc=host -u 0 --privileged --name xllm_minimax --network=host \
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
 quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-20260429
```

## 2.拉取源码并编译

下载官方仓库与模块依赖：

```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm
git checkout preview/minimax-minimal
git submodule init
git submodule update
```

下载安装依赖:

```bash
pip install --upgrade pre-commit
yum install numactl
```

执行编译，在`build/`下生成可执行文件：

```bash
python setup.py build
```

编译产物路径：`build/xllm/core/server/xllm`

## 3.启动模型

### 若机器为重启后初次拉起服务，需先执行以下脚本对device进行初始化

若不执行且npu未初始化可能导致xllm进程拉起失败

```bash
python -c "import torch_npu
for i in range(16):torch_npu.npu.set_device(i)"
```

### 环境变量

```bash
##### 1. 配置依赖路径相关环境变量
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LIBTORCH_ROOT="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"

export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/xllm/op_api/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:$LD_PRELOAD

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

##### 2. 配置日志相关环境变量
rm -rf /root/atb/log/
rm -rf /root/ascend/log/
rm -rf core.*
export ASDOPS_LOG_LEVEL=ERROR
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_TO_FILE=1

##### 3. 配置性能、通信相关环境变量
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.96
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1

export OMP_NUM_THREADS=12
export ALLOW_INTERNAL_FORMAT=1

export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0
export ATB_CONVERT_NCHW_TO_ND=1
export ATB_LAUNCH_KERNEL_WITH_TILING=1
export ATB_OPERATION_EXECUTE_ASYNC=2
export ATB_CONTEXT_WORKSPACE_SIZE=0
export INF_NAN_MODE_ENABLE=1
export HCCL_EXEC_TIMEOUT=0
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_BASE_PORT=2864
```

## 启动命令 - MiniMax-M2.7（单机 16卡 TP=16）

```bash
BATCH_SIZE=256
#推理最大batch数量
XLLM_PATH="build/xllm/core/server/xllm"
#推理入口文件路径（上一步中编译产物）
MODEL_PATH=/path/to/MiniMax-M2.7/
#模型路径

MASTER_NODE_ADDR="10.143.3.204:10015"
LOCAL_HOST="10.143.3.204"
# Service Port
START_PORT=18994
LOG_DIR="logs"
NNODES=16

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup numactl -C $((i*40))-$((i*40+39)) $XLLM_PATH \
    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$i \
    --max_memory_utilization=0.90 \
    --max_tokens_per_batch=8192 \
    --max_seqs_per_batch=$BATCH_SIZE \
    --communication_backend=hccl \
    --enable_chunked_prefill=false \
    --enable_prefix_cache=false \
    --enable_schedule_overlap=false \
    --enable_graph=false \
    --enable_atb_spec_kernel=false \
    > $LOG_FILE 2>&1 &
done
```
