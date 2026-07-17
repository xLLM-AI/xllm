---
title: "Qwen3.5"
sidebar:
  order: 1
---

+ 源码地址：https://github.com/xLLM-AI/xllm

+ 国内可用: https://gitcode.com/xLLM-AI/xllm

+ 权重下载: [modelscope-Qwen3.5-27B](https://www.modelscope.cn/models/Qwen/Qwen3.5-27B)

## 1.拉取镜像环境

首先下载xLLM提供的镜像：

```bash
# A3 arm (CANN 9)
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605
```

然后创建对应的容器

```bash
docker run -it -d \
    --ipc=host \
    -u 0 \
    --privileged \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --name xllm_qwen35 \
    --network=host \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /var/queue_schedule:/var/queue_schedule \
    -v /mnt/cfs/9n-das-admin/llm_models:/mnt/cfs/9n-das-admin/llm_models \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
    -v /var/log/npu/slog/:/var/log/npu/slog \
    -v /var/log/npu/profiling/:/var/log/npu/profiling \
    -v /var/log/npu/dump/:/var/log/npu/dump \
    -v /export/home:/export/home \
    -v ~/.ssh:/root/.ssh \
    -v /home/:/home/ \
    -v /runtime/:/runtime/ \
    -w /home \
    quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605
```

## 2.拉取源码并编译

下载官方仓库与模块依赖：

```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm
pip install pre-commit
pre-commit install
git submodule update --init --recursive
```

执行编译，在`build/`下生成可执行文件：

```bash
python setup.py build
```

编译产物路径：`build/xllm/core/server/xllm`

## 3.启动模型

### 环境变量

```bash
# 1. 配置依赖路径相关环境变量
export ASDOPS_LOG_TO_STDOUT=0
export ASDOPS_LOG_LEVEL=3
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])' | tail -n 1)"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])' | tail -n 1)"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import site, os; print(os.path.join(site.getsitepackages()[0], "torch"))')"
export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH

# 2. 加载环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_LEVEL=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.90
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=7200
export INF_NAN_MODE_ENABLE=0
export INF_NAN_MODE_FORCE_DISABLE=1

# 3. 清理旧日志
LOG_DIR="log"
mkdir -p $LOG_DIR
```

:::note
Qwen3.5 目前不支持 TP=16 场景。
:::

## 启动命令 - Qwen3.5-27B（2卡 TP=2，投机解码）

```bash
MODEL_PATH="/path/to/Qwen3.5-27B"
DRAFT_MODEL_PATH="/path/to/Qwen3.5-27B-mtp"

MASTER_NODE_ADDR="<master-host>:32764"
START_PORT=18076
NNODES=2

export ASCEND_RT_VISIBLE_DEVICES=14,15
export HCCL_IF_BASE_PORT=53433

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  ./xllm/build/xllm/core/server/xllm \
    --model $MODEL_PATH \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --max_memory_utilization=0.7 \
    --max_tokens_per_batch=32768 \
    --max_seqs_per_batch=8 \
    --block_size=128 \
    --communication_backend="lccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=true \
    --enable_schedule_overlap=true \
    --enable_graph=true \
    --node_rank=$i \
    --enable_shm=true \
    --task="generate" \
    --max_concurrent_requests=8 \
    --backend llm \
    --draft_model $DRAFT_MODEL_PATH \
    --num_speculative_tokens 3 \
    >> $LOG_FILE 2>&1 &
done
```
