---
title: "GLM-5.3-Flash Deployment"
sidebar:
  order: 3
---

+ Source code: https://github.com/xLLM-AI/xllm/tree/preview/glm-5.3-flash

+ Available in China: https://gitcode.com/xLLM-AI/xllm/tree/preview/glm-5.3-flash

+ Weight downloads:

  [ModelScope: GLM-5.3-Flash-w8a8](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-Flash-w8a8)
  
### Note
    DP, EP, prefix cache, PD disaggregation, and MTP are still under development and acceptance testing. Their performance is also being optimized. This guide currently covers only a single-node deployment without PD disaggregation.

## 1. Prepare the Image and Container

First, download an image provided by xLLM:

```bash
# A2 x86
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-x86-cann9-20260605
# A2 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-cann9-20260605
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605
```

Start a container (adjust the host-side Ascend driver, log, and model paths for your environment):

```bash
sudo docker run -it --ipc=host -u 0 --privileged --name xllm-glm53flash \
  --network=host \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /var/log/npu:/var/log/npu \
  -v /runtime:/runtime \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /export/home:/export/home \
  -v /home:/home \
  -w /export/home \
  "$IMAGE"
```

## 2. Pull the Source Code and Build

Run the following commands inside the container:

```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm
git checkout preview/glm-5.3-flash
git submodule update --init --recursive

pip install --upgrade pre-commit
yum install -y numactl
python setup.py build --device npu
```

The build artifact is `build/xllm/core/server/xllm`.

## 3. Prepare Weights and Runtime Environment

After a host reboot, initialize the NPU devices before the first service start:

```bash
python -c "import torch_npu
for i in range(8):
    torch_npu.npu.set_device(i)"
```

### Export MTP Weights

```bash
python tools/export_mtp_glm5_3_flash.py --input-dir ${W4A8/W8A8_WEIGHT_DIR} --output-dir ${EXPORTED_MTP_WEIGHT_DIR}
```

## 4. Start the Service

### Load the Ascend environment and configure runtime variables

```bash
export MODEL_PATH="/path/to/GLM-5.3-Flash-W8A8"
export DRAFT_MODEL_PATH="/path/to/GLM-5.3-Flash-W8A8-MTP"
export XLLM_PATH="/export/home/xllm/build/xllm/core/server/xllm"
```

```bash
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.95
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_BASE_PORT=47440

export GLM5_RMSNORM_ROWWISE=1
# Required when MTP is enabled

```

### Start command: 8-card single-node, GLM-5.3-Flash

```bash
LOCAL_IP=127.0.0.1
PROGRESS_CONN_PORT=9792
MASTER_NODE_ADDR="$LOCAL_IP:$PROGRESS_CONN_PORT"
START_PORT=18994
START_DEVICE=0
CORES_PER_CARD=24
NNODES=8
LOG_DIR=log
COMMUNICATION_BACKEND=hccl

mkdir -p "$LOG_DIR"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for ((i=0; i<NNODES; i++)); do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$DEVICE.log"
  nohup numactl -C $((DEVICE * CORES_PER_CARD))-$((DEVICE * CORES_PER_CARD + CORES_PER_CARD - 1)) \
    "$XLLM_PATH" \
    --model "$MODEL_PATH" \
    --model_id glm5 \
    --port "$PORT" \
    --master_node_addr="$MASTER_NODE_ADDR" \
    --nnodes="$NNODES" \
    --node_rank="$i" \
    --communication_backend="$COMMUNICATION_BACKEND" \
    --max_memory_utilization=0.75 \
    --enable_chunked_prefill=true \
    --enable_schedule_overlap=true \
    --enable_prefix_cache=false \
    --max_tokens_per_chunk_for_prefill=8192 \
    --enable_mix_batch=false \
    --enable_shm=false \
    --enable_graph=true \
    --model_impl=python \
    --backend=vlm \
    --max_seqs_per_batch=16 \
    --max_body_size=268435456 \
    --speculative_algorithm=MTP \
    --draft_model="$DRAFT_MODEL_PATH" \
    --num_speculative_tokens=1 \
    > "$LOG_FILE" 2>&1 &
done
```

Use `npu-smi info -t topo` to inspect NPU-to-CPU NUMA affinity and adjust the `numactl -C` ranges for the host topology. To disable CPU pinning, remove `numactl -C ...`.

The service is generally ready when the logs contain `Brpc Server Started`. You can also check the port directly:

```bash
curl http://127.0.0.1:18994/v1/models
```

## 5. Request Example

Text request example:

```bash
curl http://127.0.0.1:18994/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm5",
    "messages": [{"role": "user", "content": "Please introduce yourself."}],
    "stream": false,
    "max_tokens": 128
  }'
```

For GLM-5.3-Flash-VL image requests, use the OpenAI Chat Completions multimodal `content` array and replace the image placeholder with an accessible URL or data URL:

```json
{
  "model": "glm5",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image."},
      {"type": "image_url", "image_url": {"url": "<IMAGE_URL_OR_DATA_URL>"}}
    ]
  }]
}
```

## 6. Optional Debugging Environment Variables

```bash
# Deterministic computation (may affect performance)
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0

# Dynamic profiling
export PROFILING_MODE=dynamic

# Clean dynamic profiling sockets
rm -f ~/dynamic_profiling_socket_*
```

## 7. Current Limitations

- This guide covers only a single-node deployment. It does not cover multi-node communication or PD disaggregation.
- DP, EP, prefix cache, and PD disaggregation remain under development/acceptance testing and performance optimization. Refer to later release updates before enabling related options.
