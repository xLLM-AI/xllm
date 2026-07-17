---
title: "GLM-5 / GLM-5.1 / GLM-5.2"
sidebar:
  order: 2
---
+ Source code: https://github.com/xLLM-AI/xllm

+ Available in China: https://gitcode.com/xLLM-AI/xllm

+ Weight downloads:

  [modelscope-GLM-5-W8A8](https://www.modelscope.cn/models/Eco-Tech/GLM-5-W8A8-xLLM-0403/files)

  [modelscope-GLM-5.1-W8A8](https://www.modelscope.cn/models/Eco-Tech/GLM-5.1-W8A8-xLLM/files)

  [modelscope-GLM-5.1-W4A8](https://www.modelscope.cn/models/Eco-Tech/GLM-5.1-w4a8)

  [modelscope-GLM-5.2-W8A8](https://www.modelscope.cn/models/Eco-Tech/GLM-5.2-W8A8/files)

## 1. Pull the Image Environment

First, download the image provided by xLLM:

```bash
# A2 x86
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-x86-cann9-20260605
# A2 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-cann9-20260605
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605
```

**Note**: Performance stress testing has not been performed on A2 machines.

Then create the corresponding container:

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
 quay.io/jd_xllm/xllm-ai:xllm-dev-hb-rc2-x86
```

## 2. Pull the Source Code and Build

Download the official repository and module dependencies:

```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm 
git checkout release/v0.10.0
git submodule update --init --recursive
```

Download and install dependencies:

```bash
pip install --upgrade pre-commit
yum install numactl
```

Run the build. The executable `build/xllm/core/server/xllm` will be generated under `build/`:

```bash
python setup.py build --device npu
```

## 3. Start the Model

### If the service is being started for the first time after the machine has rebooted, run the following script first to initialize the devices

If this is skipped and the NPU has not been initialized, the xLLM process may fail to start.

```bash
python -c "import torch_npu
for i in range(16):torch_npu.npu.set_device(i)"
```

### Export MTP Weights

```bash
python tools/export_mtp.py --input-dir ${W4A8/W8A8_WEIGHT_DIR} --output-dir ${EXPORTED_MTP_WEIGHT_DIR}
```

### Environment Variables

```bash
##### 1. Configure environment variables
export LD_PRELOAD=/usr/lib64/libtcmalloc.so.4:$LD_PRELOAD
export HCCL_EXEC_TIMEOUT=300
export HCCL_CONNECT_TIMEOUT=300
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_BASE_PORT=2864

##### 2. Clean residual logs
rm -rf /root/ascend/log/
```

## Startup Command - A3 Single Node - GLM-5.2-W8A8

```bash
XLLM_PATH="./myxllm/xllm/build/xllm/core/server/xllm"
# Path to the xLLM executable
MODEL_PATH=/path/to/GLM-5.2-W8A8/
# Model path, using GLM-5.2-W8A8 as an example
DRAFT_MODEL_PATH=/path/to/GLM-5.2-MTP/
# MTP weights exported in the previous step

MASTER_NODE_ADDR="11.87.49.110:10015"
LOCAL_HOST="11.87.49.110"
# Service port
START_PORT=18994
START_DEVICE=0
LOG_DIR="logs"
NNODES=16

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  # Optional: bind CPU cores with numactl. Query NUMA affinity with: npu-smi info -t topo
  #nohup numactl -C $((DEVICE*40))-$((DEVICE*40+39)) $XLLM_PATH \
  nohup $XLLM_PATH \
    --model $MODEL_PATH \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$i \
    --max_memory_utilization=0.86 \
    --max_tokens_per_batch=4096 \
    --max_seqs_per_batch=16 \
    --block_size=128 \
    --enable_prefix_cache=true \
    --enable_chunked_prefill=true \
    --enable_graph=true \
    --enable_schedule_overlap=true \
    --communication_backend="hccl" \
    --graph_decode_batch_size_limit=2 \
    --draft_model=$DRAFT_MODEL_PATH \
    --num_speculative_tokens=3 \
    --ep_size=16 \
    --dp_size=2 \
    --tool_call_parser=auto \
    > $LOG_FILE 2>&1 &
done

# --max_memory_utilization   Maximum memory utilization ratio per card.
# --max_tokens_per_batch     Maximum tokens per batch. Mainly limits prefill.
# --max_seqs_per_batch       Maximum requests per batch. Mainly limits decode.
# --communication_backend    Communication backend. Options: hccl / lccl. hccl is recommended here.
# --enable_schedule_overlap  Enable async scheduling.
# --enable_prefix_cache      Enable prefix cache.
# --enable_chunked_prefill   Enable chunked prefill.
# --enable_graph             Enable aclgraph. It requires extra memory.
# --acl_graph_decode_batch_size_limit    Maximum decode batch size for graph capture. It must be <= 32 / (number of speculative tokens + 1).
# --draft_model              MTP draft-model weight path.
# --num_speculative_tokens   Number of speculative tokens predicted by MTP.
```

When the log contains `"Brpc Server Started"`, the service has started successfully.

## Other Optional Environment Variables

```bash
# Enable deterministic computation
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0

# Enable dynamic profiling mode
export PROFILING_MODE=dynamic
\rm -rf ~/dynamic_profiling_socket_*
```

## Startup Command - Two-Machine Startup Example

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

for (( i=0; i<$LOCAL_NODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup numactl -C $((DEVICE*40))-$((DEVICE*40+39)) $XLLM_PATH \
    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$i \
    --max_memory_utilization=0.85 \
    --max_tokens_per_batch=8192 \
    --max_seqs_per_batch=128 \
    --block_size=128 \
    --enable_prefix_cache=true \
    --enable_chunked_prefill=true \
    --communication_backend="hccl" \
    --enable_schedule_overlap=true \
    --enable_graph=true \
    --acl_graph_decode_batch_size_limit=4 \
    --draft_model=$DRAFT_MODEL_PATH \
    --num_speculative_tokens=3 \
    --ep_size=32 \
    --dp_size=4 \
    --rank_tablefile=/yourPath/ranktable.json \
    --tool_call_parser=auto \
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

for (( i=0; i<$LOCAL_NODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup numactl -C $((DEVICE*40))-$((DEVICE*40+39)) $XLLM_PATH \
    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$((i + LOCAL_NODES)) \
    --max_memory_utilization=0.85 \
    --max_tokens_per_batch=8192 \
    --max_seqs_per_batch=128 \
    --block_size=128 \
    --enable_prefix_cache=true \
    --enable_chunked_prefill=true \
    --communication_backend="hccl" \
    --enable_schedule_overlap=true \
    --enable_graph=true \
    --acl_graph_decode_batch_size_limit=4 \
    --draft_model=$DRAFT_MODEL_PATH \
    --num_speculative_tokens=3 \
    --ep_size=32 \
    --dp_size=4 \
    --rank_tablefile=/yourPath/ranktable.json \
    --tool_call_parser=auto \
    > $LOG_FILE 2>&1 &
done
```

### Rank Table Examples

[A3 rank table configuration](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/hcclug/hcclug_000066.html)

[A2 rank table configuration](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/hcclug/hcclug_000067.html)

Note that the rank table formats differ between A3 and A2.

## View Device NUMA Affinity

Command:

```bash
npu-smi info -t topo
```

In the preceding commands:

```bash
numactl -C $((DEVICE*12))-$((DEVICE*12+11))
```

indicates that the process is bound to the corresponding affinity cores. You can modify the bound core IDs according to the machine.

## EX3. GLM-5 Weight Quantization

GLM-5.2 quantization guidance will be updated later.

### Install msmodelslim

```bash
pip install transformers==5.2.0

git clone https://gitcode.com/Ascend/msmodelslim.git
cd msmodelslim
bash install.sh
```

### Run Quantization

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu:0 \
  --model_type GLM-5 \
  --quant_type w8a8 \
  --trust_remote_code True
```

## PD Disaggregation

### Install etcd and xllm-service

#### PD Disaggregated Deployment

`xllm` supports PD disaggregated deployment. This must be used together with another open-source library, [xllm service](https://github.com/xLLM-AI/xllm-service).

##### xLLM Service Dependencies

First, download and install `xllm service`, similar to installing and building `xllm`:

```bash
git clone https://github.com/xLLM-AI/xllm-service.git
cd xllm-service
git submodule init
git submodule update
```

##### Install etcd

`xllm_service` depends on [etcd](https://github.com/etcd-io/etcd). Use the official etcd [installation script](https://github.com/etcd-io/etcd/releases) to install it. The default installation path used by the script is `/tmp/etcd-download-test/etcd`. You can manually modify the installation path in the script, or move it manually after the script finishes:

```bash
mv /tmp/etcd-download-test/etcd /path/to/your/etcd
```

##### Build xLLM Service

Apply the patch first:

```bash
sh prepare.sh
```

Then build:

```bash
mkdir -p build
cd build
cmake ..
make -j 8
cd ..
```

:::caution[Possible Errors]
You may encounter installation errors related to `boost-locale` and `boost-interprocess`: `vcpkg-src/packages/boost-locale_x64-linux/include: No such     file or directory`, `/vcpkg-src/packages/boost-interprocess_x64-linux/include: No such file or directory`.
Reinstall these packages with `vcpkg`:
```bash
/path/to/vcpkg remove boost-locale boost-interprocess
/path/to/vcpkg install boost-locale:x64-linux
/path/to/vcpkg install boost-interprocess:x64-linux
```

:::

### Run PD Disaggregation

Start etcd:

```bash
./etcd-download-test/etcd --listen-peer-urls 'http://localhost:2390'  --listen-client-urls 'http://localhost:2389' --advertise-client-urls  'http://localhost:2391'
```

For cross-machine configuration, refer to the following etcd command:

```bash
/tmp/etcd-download-test/etcd --listen-peer-urls 'http://0.0.0.0:3390' --listen-client-urls 'http://0.0.0.0:3389' --advertise-client-urls 'http://11.87.191.82:3389'
```

Start xllm service:

```bash
ENABLE_DECODE_RESPONSE_TO_SERVICE=true ./xllm_master_serving --etcd_addr="127.0.0.1:12389" --http_server_port 28888 --rpc_server_port 28889 --tokenizer_path=/export/home/models/GLM-5-W8A8/
```

For cross-machine configuration, start xllm service with:

```bash
ENABLE_DECODE_RESPONSE_TO_SERVICE=true ../xllm-service/build/xllm_service/xllm_master_serving --etcd_addr="11.87.191.82:3389" --http_server_port 38888 --rpc_server_port 38889 --tokenizer_path=/export/home/models/GLM-5-W8A8/
```

- Start the Prefill instance
```bash
  BATCH_SIZE=256
  # Maximum inference batch size
  XLLM_PATH="./myxllm/xllm/build/xllm/core/server/xllm"
  # Inference entry binary path, which is the build artifact from the previous step
  MODEL_PATH=/export/home/models/GLM-5-w8a8/
  # Model path, here using the int-quantized GLM-5
  DRAFT_MODEL_PATH=/export/home/models/GLM-5-MTP/
  
  MASTER_NODE_ADDR="11.87.49.110:10015"
  LOCAL_HOST="11.87.49.110"
  # Service port
  START_PORT=18994
  START_DEVICE=0
  LOG_DIR="logs"
  NNODES=16
  
  for (( i=0; i<$NNODES; i++ ))
  do
    PORT=$((START_PORT + i))
    DEVICE=$((START_DEVICE + i))
    LOG_FILE="$LOG_DIR/node_$i.log"
    nohup numactl -C $((i*40))-$((i*40+39)) $XLLM_PATH \
      --model $MODEL_PATH  --model_id glmmoe \
      --host $LOCAL_HOST \
      --port $PORT \
      --master_node_addr=$MASTER_NODE_ADDR \
      --nnodes=$NNODES \
      --node_rank=$i \
      --max_memory_utilization=0.86 \
      --max_tokens_per_batch=5000 \
      --max_seqs_per_batch=$BATCH_SIZE \
      --communication_backend=hccl \
      --enable_schedule_overlap=true \
      --enable_prefix_cache=false \
      --enable_chunked_prefill=false \
      --enable_graph=true \
      --draft_model $DRAFT_MODEL_PATH \
      --num_speculative_tokens 1 \
      --tool_call_parser=auto \
      --enable_disagg_pd=true \
      --instance_role=PREFILL \
      --etcd_addr=$LOCAL_HOST:3389 \
      --transfer_listen_port=$((36100 + i)) \
      --disagg_pd_port=8877 \
      > $LOG_FILE 2>&1 &
  done
  
  # --etcd_addr=$LOCAL_HOST:3389  Refer to the advertise-client-urls configuration in etcd.
  # --instance_role=DECODE        PD configuration: DECODE or PREFILL.
  ```

- Start the Decode instance
  
  ```bash
    BATCH_SIZE=256
  # Maximum inference batch size
  XLLM_PATH="./myxllm/xllm/build/xllm/core/server/xllm"
  # Inference entry binary path, which is the build artifact from the previous step
  MODEL_PATH=/export/home/models/GLM-5-w8a8/
  # Model path, here using the int-quantized GLM-5
  DRAFT_MODEL_PATH=/export/home/models/GLM-5-MTP/
  
  MASTER_NODE_ADDR="11.87.49.110:10015"
  LOCAL_HOST="11.87.49.110"
  # Service port
  START_PORT=18994
  START_DEVICE=0
  LOG_DIR="logs"
  NNODES=16
  
  for (( i=0; i<$NNODES; i++ ))
  do
    PORT=$((START_PORT + i))
    DEVICE=$((START_DEVICE + i))
    LOG_FILE="$LOG_DIR/node_$i.log"
    nohup numactl -C $((i*40))-$((i*40+39)) $XLLM_PATH \
      --model $MODEL_PATH  --model_id glmmoe \
      --host $LOCAL_HOST \
      --port $PORT \
      --master_node_addr=$MASTER_NODE_ADDR \
      --nnodes=$NNODES \
      --node_rank=$i \
      --max_memory_utilization=0.86 \
      --max_tokens_per_batch=5000 \
      --max_seqs_per_batch=$BATCH_SIZE \
      --communication_backend=hccl \
      --enable_schedule_overlap=true \
      --enable_prefix_cache=false \
      --enable_chunked_prefill=false \
      --enable_graph=true \
      --draft_model $DRAFT_MODEL_PATH \
      --num_speculative_tokens 1 \
      --tool_call_parser=auto \
      --enable_disagg_pd=true \
      --instance_role=DECODE \
      --etcd_addr=$LOCAL_HOST:3389 \
      --transfer_listen_port=$((36100 + i)) \
      --disagg_pd_port=8877 \
      > $LOG_FILE 2>&1 &
  done
  
  # --etcd_addr=$LOCAL_HOST:3389  Refer to the advertise-client-urls configuration in etcd.
  # --instance_role=DECODE        PD configuration: DECODE or PREFILL.
  ```
  
  Notes:

- PD disaggregation needs to read `/etc/hccn.conf`. Make sure this file on the physical machine is mounted into the container.

- `etcd_addr` must be the same as the `etcd_addr` used by `xllm_service`.
  The test command is similar to the one above. Note that for `curl http://localhost:{PORT}/v1/chat/completions ...`, `PORT` should be the `http_server_port` used to start xLLM service.

- When deploying P or Q across multiple machines, such as deploying two P instances, add `--rank_tablefile` to complete communication.
