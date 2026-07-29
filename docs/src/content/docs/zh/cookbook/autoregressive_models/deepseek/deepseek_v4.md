---
title: "DeepSeek-V4"
description: "DeepSeek-V4 在 Ascend A3 设备上的 xLLM 推理实践指南"
---
# 使用 xLLM 在 Ascend A3 设备 推理

源码地址：https://github.com/jd-opensource/xllm

国内可用: https://gitcode.com/xLLM-AI/xllm

权重下载

Flash权重：
https://modelers.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp

Pro权重:
https://modelers.cn/models/Eco-Tech/DeepSeek-V4-Pro-w4a8-mtp


## 1. 拉取镜像环境

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

## 2. 拉取源码并编译

下载官方仓库与模块依赖：

```bash
git clone https://github.com/jd-opensource/xllm
cd xllm 
git submodule update --init --recursive
```

下载安装依赖:

```bash
pip install --upgrade pre-commit
```

执行编译，在`build/`下生成可执行文件`build/xllm/core/server/xllm`：

```bash
python setup.py build --device npu
```

## 3. 启动模型

### 若机器为重启后初次拉起服务，需先执行以下脚本对device进行初始化

> 若不执行且 npu 未初始化可能导致 xllm 进程拉起失败

```bash
python -c "import torch_npu
for i in range(16):torch_npu.npu.set_device(i)"
```

### 导出MTP权重

```bash
python tools/export_mtp.py --input-dir ${W4A8/W8A8权重目录} --output-dir ${导出MTP权重目录}
```

### 环境变量

```bash
##### 1， 配置依赖路径相关环境变量

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source ${ASCEND_TOOLKIT_HOME}/opp/vendors/custom_xllm_math/bin/set_env.bash

##### 2， 配置日志相关环境变量
rm -rf /root/ascend/log/
rm -rf core.*

##### 3. 配置性能、通信相关环境变量
export HCCL_IF_BASE_PORT=43432
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.96
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export ATB_CONTEXT_WORKSPACE_SIZE=0
export OMP_NUM_THREADS=12
export ALLOW_INTERNAL_FORMAT=1

```

## 启动命令 - 单机拉起样例

```bash
BATCH_SIZE=256
#推理最大batch数量
XLLM_PATH="./myxllm/xllm/build/xllm/core/server/xllm"
#推理入口文件路径（上一步中编译产物）
MODEL_PATH=/path/to/dsv4
#模型路径
DRAFT_MODEL_PATH=/path/to/dsv4_mtp
#导出的mtp权重

MASTER_NODE_ADDR="11.87.49.110:10015"
LOCAL_HOST="11.87.49.110"
# Service Port
START_PORT=18994
LOG_DIR="logs"
NNODES=8

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup $XLLM_PATH -model-id ds \
    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$i \
    --max_memory_utilization=0.9 \
    --max_tokens_per_batch=2048 \
    --max_seqs_per_batch=32 \
    --block_size=128 \
    --communication_backend="hccl" \
    --tool_call_parser=deepseekv4 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=true \
    --enable_schedule_overlap=true \
    --enable_graph=true \
    --npu_kernel_backend=TORCH \
    --ep_size=8 \
    --dp_size=2 \
    > $LOG_FILE 2>&1 &
done

    # 开启mtp时需要的变量
    # --draft_model=$DRAFT_MODEL_PATH \
    # --num_speculative_tokens=1 \

# numactl -C xxxxx          亲和性绑核(NUMA亲和性查询命令： npu-smi info -t topo)
#--max_memory_utilization   单卡最大显存占用比例
#--max_tokens_per_batch     单batch最大token数  （主要限制prefill）
#--max_seqs_per_batch       单batch最大请求数   （主要限制decoe）
#--communication_backend    通信backend 可选(hccl / lccl) 此处建议hccl
#--enable_schedule_overlap  开启异步调度
#--enable_prefix_cache      开启prefix_cache
#--enable_chunked_prefill   开启chunked_prefill
#--enable_graph             开启aclgraph
#--draft_model              mtp - mtp权重路径
#--num_speculative_tokens   mtp - 预测token数
```

日志出现"Brpc Server Started"表示服务成功拉起。

## 其他可选环境变量

```bash
#开启确定性计算
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0

# #开启动态profiling模式
# export PROFILING_MODE=dynamic
# \rm -rf ~/dynamic_profiling_socket_*
```

## 启动命令 - 双机拉起样例

### Node0 (master)

```bash
MASTER_NODE_ADDR="11.87.49.110:19990"
LOCAL_HOST="11.87.49.110"
START_PORT=15890
LOG_DIR="logs"
NNODES=32
LOCAL_NODES=16
export HCCL_IF_BASE_PORT=48439
unset HCCL_OP_EXPANSION_MODE

for (( i=0; i<$LOCAL_NODES; i++ )); do
  PORT=$((START_PORT + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup $XLLM_PATH \
    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$i \
    ......
    --rank_tablefile=/yourPath/ranktable.json \
    > $LOG_FILE 2>&1 &
done
```

#### Node1 (worker)

```bash
MASTER_NODE_ADDR="11.87.49.110:19990"
LOCAL_HOST="11.87.49.111"
START_PORT=15890
LOG_DIR="logs"
NNODES=32
LOCAL_NODES=16
export HCCL_IF_BASE_PORT=48439
unset HCCL_OP_EXPANSION_MODE

for (( i=0; i<$LOCAL_NODES; i++ )); do
  PORT=$((START_PORT + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup  $XLLM_PATH \
    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$((i + LOCAL_NODES)) \
    ......
    --rank_tablefile=/yourPath/ranktable.json \
    > $LOG_FILE 2>&1 &
done
```

### ranktable样例

 [A3 ranktable配置](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/hcclug/hcclug_000066.html)

 [A2 ranktable配置](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/hcclug/hcclug_000067.html)

 （注意A3与A2的ranktable格式差异）
