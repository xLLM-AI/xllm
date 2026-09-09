---
title: "GLM-5.3-Flash 部署"
sidebar:
  order: 3
---

+ 源码地址：https://github.com/xLLM-AI/xllm/tree/preview/glm-5.3-flash

+ 国内可用: https://gitcode.com/xLLM-AI/xllm/tree/preview/glm-5.3-flash

+ 权重下载:

  [ModelScope：GLM-5.3-Flash-w8a8](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-Flash-w8a8)
  
### 注意
    DP、EP、prefix cache、PD分离、MTP 等特性仍在开发/验收中，相关性能也在持续优化。本文暂时仅提供单机、非 PD 分离的部署方式。

## 1. 准备镜像和容器

首先下载xLLM提供的镜像：

```bash
# A2 x86
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-x86-cann9-20260605
# A2 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-cann9-20260605
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605
```

启动容器（宿主机上的 Ascend 驱动、日志、模型目录按实际环境调整）：

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

## 2.拉取源码并编译

在容器中执行：

```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm
git checkout preview/glm-5.3-flash
git submodule update --init --recursive

pip install --upgrade pre-commit
yum install -y numactl
python setup.py build --device npu
```

编译产物为 `build/xllm/core/server/xllm`。

## 3. 准备权重和运行环境

机器重启后首次启动服务时，先初始化 NPU device：

```bash
python -c "import torch_npu
for i in range(8):
    torch_npu.npu.set_device(i)"
```

### 导出MTP权重

```bash
python tools/export_mtp_glm5_3_flash.py --input-dir ${W4A8/W8A8权重目录} --output-dir ${导出MTP权重目录}
```

## 4. 启动服务

### 加载 Ascend 环境并设置运行参数：

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
#开启mtp时需要加这个参数

```

### 启动命令 - 8Node 单机 - GLM-5.3-Flash

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
    --draft_model=$DRAFT_MODEL_PATH \
    --num_speculative_tokens=1 \
    > "$LOG_FILE" 2>&1 &
done
```

可使用 `npu-smi info -t topo` 查看 NPU 与 CPU NUMA 亲和性，并按机器拓扑调整 `numactl -C` 的核范围。若不需要绑核，可移除 `numactl -C ...`。

日志中出现 `Brpc Server Started` 后，服务通常已完成启动。也可以检查端口：

```bash
curl http://127.0.0.1:18994/v1/models
```

## 5. 调用示例

文本请求示例：

```bash
curl http://127.0.0.1:18994/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm5",
    "messages": [{"role": "user", "content": "请介绍一下你自己。"}],
    "stream": false,
    "max_tokens": 128
  }'
```

GLM-5.3-Flash-VL 的图像请求请按照 OpenAI Chat Completions 的多模态格式传入 `content` 数组，并将图片替换为可访问的 URL 或 data URL：

```json
{
  "model": "glm5",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "描述这张图片。"},
      {"type": "image_url", "image_url": {"url": "<IMAGE_URL_OR_DATA_URL>"}}
    ]
  }]
}
```

## 6. 可选调试环境变量

```bash
# 确定性计算（会影响性能）
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0

# 动态 profiling
export PROFILING_MODE=dynamic

# 动态 profiling socket 清理
rm -f ~/dynamic_profiling_socket_*
```

## 7. 当前限制

- 本文只覆盖单机启动，不覆盖多机通信 / PD 分离。
- DP、EP、prefix cache、PD 分离仍处于开发/验收和性能优化阶段，相关参数请等待后续版本更新。
