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

带 DSpark 权重的 DeepSeek-V4-Flash-0731 W8A8：
https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8

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

### 选择 speculative decoding 权重格式

原有 DeepSeek-V4 MTP 路径和 DeepSeek-V4-Flash-0731 DSpark 路径使用不同的
draft 模型格式：

- 使用原有 MTP 路径时，需要将 MTP 权重额外导出到独立目录：

```bash
python tools/export_mtp.py --input-dir ${W4A8/W8A8权重目录} --output-dir ${导出MTP权重目录}
```

- 使用 DeepSeek-V4-Flash-0731 DSpark 时，**不要**执行 `export_mtp.py`。
  三层 DSpark、独立的词表 embedding/head 和 Markov head 都保留在 target
  权重的 `mtp.0`、`mtp.1`、`mtp.2` 下。`--model` 和 `--draft_model` 应指向
  同一个原始或量化后的 0731 权重目录。原始 FP 权重可能复用顶层
  `embed.weight/head.weight`，QuaRot 权重可能带独立的
  `mtp.0.embed.weight/mtp.2.head.weight`；xLLM 同时兼容两种格式，且两者
  同时存在时优先使用 DSpark 独立词表权重。

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

    # DeepSeek-V4-Flash-0731 DSpark 改用：
    # --speculative_algorithm=DSpark \
    # --draft_model=$MODEL_PATH \
    # --num_speculative_tokens=5 \

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

### DSpark 使用方式

DSpark 不需要额外导出 MTP 权重。将 `--model` 和 `--draft_model` 设置为同一个
DeepSeek-V4-Flash-0731 权重目录即可：

```bash
--speculative_algorithm=DSpark \
--model=/path/to/DeepSeek-V4-Flash-0731-w8a8 \
--draft_model=/path/to/DeepSeek-V4-Flash-0731-w8a8 \
--num_speculative_tokens=5
```

推荐使用 `--num_speculative_tokens=5`，因为 0731 权重按
`dspark_block_size=5` 训练。改用其他 gamma 会改变扩散块几何，超出训练分布，
需要重新验证接受率和性能。当前路径暂不支持 `cp_size > 1`。

在 NPU 上，xLLM 支持两种 SAS 模式。默认兼容模式适配 CANN 9.0，无需增加参数；
若当前 SAS 算子支持非空 `ori_sparse_indices`，可设置
`--enable_dspark_native_sas=true`，使用完整的 DSpark SWA 窗口。旧版算子会在
tiling 阶段直接终止进程，因此无法安全地自动探测该能力。

可通过逐位置计数观察 DSpark 接受率：

```bash
curl http://${HOST}:${PORT}/brpc_metrics | grep speculative_num
```

```text
acceptance[position] =
  speculative_num_accepted_tokens_per_pos{position} /
  speculative_num_drafts_total
```

逐位置值表示“接受前缀能到达该位置”的概率。指标
`speculative_mean_tokens_per_decode_step` 现在表示每个 proposal 序列累计平均
提交 token 数，由
`speculative_num_committed_tokens_total / speculative_num_drafts_total` 计算。

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
