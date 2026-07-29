---
title: "Wan2.2 Deployment"
sidebar:
  order: 2
---

This document describes how to deploy the Wan2.2 video generation service using xLLM in an Ascend NPU environment.

## 1. Pull the Docker Image

First, pull the xLLM-provided image:

```bash
# A2 x86
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-x86-cann9-20260605
# A2 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-cann9-20260605
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605
```

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
 quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605
```

## 2. Clone the Source Code and Build

Clone the official repository and module dependencies:

```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm
git submodule update --init --update
```

Install dependencies:

```bash
pip install --upgrade pre-commit
```

Build the project to generate the executable `build/xllm/core/server/xllm` under the `build/` directory:

```bash
python setup.py build --device npu
```

## 3. Model Weight Preparation

The model root directory should contain the component directories required by the DiT service. A typical directory structure is as follows:

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

## 4. Start the Model and Expose the `/v1/video/generation` API

### Kill Previous xLLM Service Processes

```bash
pkill -9 xllm
```

### Environment Variables

```bash
# 0. Load Ascend environment (must be executed before python3)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

# 1. Configure dependency path environment variables
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH

# 2. NPU-device environment variables
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

# 3. Clean up old logs
\rm -rf core.*
\rm -rf log/node_*.log
```

## Startup Command — Single-Machine Example

```bash
# 4. Inference configuration
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
```

When "Brpc Server Started" appears in the log, the service has started successfully.

## DiT Parameter Reference

| Parameter | Description | Default | Values |
| --------- | ----------- | ------- | ------ |
| `--sp_size` | Sequence Parallel degree | `1` | Positive integer, e.g. `1`, `2`, `4`, `8` |
| `--cfg_size` | Classifier-Free Guidance parallel degree | `1` | `1` or `2` |
| `--vae_size` | VAE Spatial Parallel degree | `1` | Positive integer, e.g. `1`, `2`, `4`; can be set to the same value as `sp_size` |
| `--tp_size` | Tensor Parallel degree | `1` | Positive integer, e.g. `2`, `4`; it is recommended not to enable this and use dynamic weight loading instead |
| `--enable_rolling_load` | Enable dynamic weight loading | `false` | Bool, `true` or `false` |
| `--rolling_load_num_rolling_slot` | Number of slots for dynamic weight loading | `2` | Positive integer, e.g. `2`, `3` |
| `--dit_laser_attention_enable` | Enable Laser Attention | `false` | Bool, `true` or `false` |
| `--dit_distill_enable` | Enable distilled model | `false` | Bool, `true` or `false` |
| `--dit_sparse_attention_enabled` | Enable sparse attention | `false` | Bool, `true` or `false`; mutually exclusive with `laser_attention` |
| `--dit_sparse_attention_sparsity` | Sparsity ratio for sparse attention | `0.5` | Float, e.g. `0.5`, `0.6`; depends on `dit_sparse_attention_enabled` |
| `--dit_sparse_attention_sparse_start_step` | Step at which sparsification starts | `0` | Integer, e.g. `5`, `10`; depends on `dit_sparse_attention_enabled` |
| `--dit_sparse_attention_version` | Sparse attention version | `rain_fusion` | String, `rain_fusion` or `sparse_attention`; depends on `dit_sparse_attention_enabled` |

`NNODES` must equal `sp_size * cfg_size * tp_size`.

| `sp_size` | `cfg_size` | `vae_size` | `tp_size` | `NNODES` | Description |
| --------- | ---------- | ---------- | --------- | -------- | ----------- |
| `1` | `1` | `1` | `1` | `1` | Single-card deployment |
| `2` | `1` | `1` | `1` | `2` | SP only |
| `1` | `2` | `1` | `1` | `2` | CFG parallel only |
| `2` | `2` | `2` | `1` | `4` | SP + CFG + VAE parallel |
| `4` | `2` | `2` | `1` | `8` | 8-card deployment |
