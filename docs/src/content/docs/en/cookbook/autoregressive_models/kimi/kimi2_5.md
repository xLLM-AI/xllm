---
title: "Kimi-K2.5 / Kimi-K2.6"
sidebar:
  order: 2
---

- Source code: [https://github.com/xLLM-AI/xllm](https://github.com/xLLM-AI/xllm)
- Available in China: [https://gitcode.com/xLLM-AI/xllm](https://gitcode.com/xLLM-AI/xllm)
- Kimi-K2.5 W8A8 weight download: [modelscope-Kimi-K2.5-W8A8-xLLM](https://www.modelscope.cn/models/Eco-Tech/Kimi-K2.5-W8A8-xLLM)
- Kimi-K2.6 W8A8 weight download: [modelscope-Kimi-K2.6-w8a8-xllm](https://www.modelscope.cn/models/Eco-Tech/Kimi-K2.6-w8a8-xllm)

P.S. Kimi-K2.5 and Kimi-K2.6 use the same model architecture. The following sections use Kimi-K2.5 as an example to describe the overall deployment process.

## 0. Weight Preparation

### Download Weights from ModelScope

```bash
export MODELSCOPE_CACHE=path-to-model # Default: ~/.cache/modelscope/hub
pip install modelscope
modelscope download --model Eco-Tech/Kimi-K2.5-W8A8-xLLM
```

## 1. Pull the Image Environment

First, download the image provided by xLLM:

```bash
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-20260429
```

Then create the corresponding container:

```bash
sudo docker run -it --ipc=host -u 0 --privileged --name xllm_kimi_k25 --network=host \
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

## 2. Pull the Source Code and Build

Download the official repository and module dependencies:

```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm
git checkout main
git submodule init
git submodule update
```

Download and install dependencies:

```bash
pip install --upgrade pre-commit
yum install numactl
```

Run the build to generate the executable under `build/`:

```bash
python setup.py build
```

Build artifact path: `build/xllm/core/server/xllm`

## 3. Start the Model

### If the service is being started for the first time after the machine has rebooted, run the following script first to initialize the devices

If this is skipped and the NPU has not been initialized, the xLLM process may fail to start.

```bash
python -c "import torch_npu
for i in range(16):torch_npu.npu.set_device(i)"
```

### Environment Variables

```bash
##### 1. Configure dependency path environment variables
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

##### 2. Configure log-related environment variables
rm -rf /root/atb/log/
rm -rf /root/ascend/log/
rm -rf core.*
export ASDOPS_LOG_LEVEL=ERROR
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_TO_FILE=1

##### 3. Configure performance and communication-related environment variables
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

## Startup Command - Kimi_k25 (two machines, 16 cards, 32 dies, tp=4, dp=8, ep=32)

### Node0 (master)

```bash
MASTER_NODE_ADDR="11.87.49.110:19990"
LOCAL_HOST="11.87.49.110"
START_PORT=15890
START_DEVICE=0
LOG_DIR="logs"
NNODES=32
LOCAL_NODES=16
export HCCL_IF_BASE_PORT=48439
unset HCCL_OP_EXPANSION_MODE

for (( i=0; i<$LOCAL_NODES; i++ ))do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i));  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup numactl -C $((DEVICE*40))-$((DEVICE*40+39)) $XLLM_PATH \
    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$i \
    --max_memory_utilization=0.85 \
    --max_tokens_per_batch=8192 \
    --max_seqs_per_batch=20 \
    --block_size=128 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --communication_backend="hccl" \
    --enable_schedule_overlap=true \
    --enable_graph=false \
    --enable_shm=true \
    --ep_size=32 \
    --dp_size=8 \
    --input_shm_size=4096 \
    --rank_tablefile=/yourPath/ranktable.json \
    > $LOG_FILE 2>&1 &
done
```

#### Node1 (worker)

```bash
MASTER_NODE_ADDR="11.87.49.110:19990"
LOCAL_HOST="11.87.49.111"
START_PORT=15890
START_DEVICE=0
LOG_DIR="logs"
NNODES=32
LOCAL_NODES=16
export HCCL_IF_BASE_PORT=48439
unset HCCL_OP_EXPANSION_MODE

for (( i=0; i<$LOCAL_NODES; i++ ))do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i));  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup numactl -C $((DEVICE*40))-$((DEVICE*40+39)) $XLLM_PATH \
    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$((i + LOCAL_NODES)) \
    --max_memory_utilization=0.85 \
    --max_tokens_per_batch=8192 \
    --max_seqs_per_batch=20 \
    --block_size=128 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --communication_backend="hccl" \
    --enable_schedule_overlap=true \
    --enable_graph=false \
    --enable_shm=true \
    --ep_size=32 \
    --dp_size=8 \
    --input_shm_size=4096 \
    --rank_tablefile=/yourPath/ranktable.json \
done
```

#### ranktable Example

ranktable configuration guide: [https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/hccl/hcclug/hcclug_000014.html](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/hccl/hcclug/hcclug_000014.html)

```bash
ln -s /usr/local/Ascend/driver/tools/hccn_tool /usr/sbin/

#device_ip
for i in {0..15};do hccn_tool -i $i -vnic -g; done

#super_device_id
for i in {0..7};do for j in {0..1}; do npu-smi info -t spod-info -i $i -c $j; done; done
```

```json
{
    "status": "completed",
    "version": "1.2",
    "server_count": "2",
    "server_list": [
        {
            "server_id": "10.87.191.98",
            "host_nic_ip": "reserve",
            "host_ip": "10.87.191.98",
            "container_ip": "10.87.191.98",
            "device": [
                {
                    "device_id": "0",
                    "device_ip": "192.24.2.199",
                    "super_device_id": "100663296",
                    "rank_id": "16"
                },
                ...
                {
                    "device_id": "15",
                    "device_ip": "192.24.3.184",
                    "super_device_id": "102563855",
                    "rank_id": "31"
                }
            ]
        },
        {
            "server_id": "10.87.191.102",
            "host_nic_ip": "reserve",
            "host_ip": "10.87.191.102",
            "container_ip": "10.87.191.102",
            "device": [
                {
                    "device_id": "0",
                    "device_ip": "192.28.2.199",
                    "super_device_id": "117440512",
                    "rank_id": "0"
                },
                ...
                {
                    "device_id": "15",
                    "device_ip": "192.28.3.184",
                    "super_device_id": "119341071",
                    "rank_id": "15"
                }
            ]
        }
    ],
    "super_pod_list": [
        {
            "super_pod_id": "2",
            "server_list": [
                {
                    "server_id": "10.87.191.98"
                },
                {
                    "server_id": "10.87.191.102"
                }
            ]
        }
    ]
}
```

When the log contains `"Application startup complete."`, the service has started successfully.
