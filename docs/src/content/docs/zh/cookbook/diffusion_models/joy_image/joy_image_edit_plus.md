---
title: "JoyAI-Image Edit Plus 部署"
description: "基于 xLLM 和 Ascend NPU 部署 JoyAI-Image Edit Plus 图片编辑服务"
sidebar:
  order: 1
---

本文档介绍如何基于 xLLM 在 Ascend NPU 环境中部署
JoyAI-Image Edit Plus 图片编辑服务。该模型支持输入多张参考图片和文本指令，
通过 `/v1/image/generation` 接口返回编辑后的图片。

本文只介绍图片生成业务，不涉及视频生成接口或视频相关参数。

- xLLM 源码：https://github.com/xLLM-AI/xllm
- JoyAI-Image 源码：https://github.com/jd-opensource/JoyAI-Image
- 模型权重：https://huggingface.co/jdopensource/JoyAI-Image-Edit-Plus-Diffusers

## 1. 拉取镜像环境

首先下载 xLLM 提供的镜像：

```bash
# A2 x86
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-x86-cann9-20260605
# A2 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-cann9-20260605
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605
```

以下命令以 A3 arm 镜像为例创建容器：

```bash
IMAGE=quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605
CONTAINER=xllm-joy-image-edit-plus

docker run -it --ipc=host -u 0 --privileged \
  --name "${CONTAINER}" \
  --network=host \
  -v /var/queue_schedule:/var/queue_schedule \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
  -v /var/log/npu/slog/:/var/log/npu/slog \
  -v /var/log/npu/profiling/:/var/log/npu/profiling \
  -v /var/log/npu/dump/:/var/log/npu/dump \
  -v /runtime/:/runtime/ \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /export/home:/export/home \
  -v /home/:/home/ \
  -w /export/home \
  "${IMAGE}"
```

## 2. 拉取源码并编译

下载官方仓库及子模块：

```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm
git submodule update --init --recursive
```

安装构建依赖并编译：

```bash
pip install --upgrade pre-commit
python setup.py build
```

编译完成后，可执行文件位于：

```text
build/xllm/core/server/xllm
```

## 3. 准备模型权重

下载 JoyAI-Image Edit Plus 的 Diffusers 格式权重，并将其保存到所有服务进程
都可以访问的目录。模型根目录需要包含以下组件：

```text
joyai-image-edit-plus/
├── model_index.json
├── processor/
├── scheduler/
├── text_encoder/
├── tokenizer/
├── transformer/
└── vae/
```

`model_index.json` 中的流水线类型应为 `JoyImageEditPlusPipeline`。xLLM 会在
同一个 DiT 服务中加载 Qwen3-VL text encoder、JoyImage transformer 和 VAE，
因此不需要额外启动独立的 embedding 服务。

### 3.1 准备 text encoder 配置

xLLM 会从 `text_encoder/` 组件目录加载 Qwen3-VL 的模型配置、Tokenizer 和
多模态 Processor 配置。原始权重中的 Tokenizer 文件位于 `processor/` 目录，
启动服务前需要将它们复制到 `text_encoder/`：

```bash
MODEL_PATH=/export/home/models/joyai-image-edit-plus

cp "${MODEL_PATH}/processor/tokenizer_config.json" \
  "${MODEL_PATH}/text_encoder/tokenizer_config.json"
cp "${MODEL_PATH}/processor/tokenizer.json" \
  "${MODEL_PATH}/text_encoder/tokenizer.json"
```

其中 `tokenizer.json` 包含完整词表，必须复制原始文件，不能使用精简配置代替。
此外，`text_encoder/` 下还需要准备 `config.json`、
`preprocessor_config.json` 和 `video_preprocessor_config.json`。最终目录至少应
包含以下文件：

```text
text_encoder/
├── config.json
├── model-00001-of-*.safetensors
├── model.safetensors.index.json
├── preprocessor_config.json
├── tokenizer.json
├── tokenizer_config.json
└── video_preprocessor_config.json
```

`text_encoder/config.json` 配置如下：

```json
{
  "architectures": [
    "Qwen3VLForConditionalGeneration"
  ],
  "dtype": "bfloat16",
  "image_token_id": 151655,
  "model_type": "qwen3_vl",
  "text_config": {
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "dtype": "bfloat16",
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 12288,
    "max_position_embeddings": 262144,
    "model_type": "qwen3_vl_text",
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "pad_token_id": null,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
      "mrope_interleaved": true,
      "mrope_section": [
        24,
        20,
        20
      ],
      "rope_theta": 5000000,
      "rope_type": "default"
    },
    "rope_theta": 5000000,
    "use_cache": true,
    "vocab_size": 151936
  },
  "tie_word_embeddings": false,
  "transformers_version": "5.6.0",
  "video_token_id": 151656,
  "vision_config": {
    "deepstack_visual_indexes": [
      8,
      16,
      24
    ],
    "depth": 27,
    "dtype": "bfloat16",
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 1152,
    "in_channels": 3,
    "initializer_range": 0.02,
    "intermediate_size": 4304,
    "model_type": "qwen3_vl_vision",
    "num_heads": 16,
    "num_position_embeddings": 2304,
    "out_hidden_size": 4096,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2
  },
  "vision_end_token_id": 151653,
  "vision_start_token_id": 151652
}
```

`text_encoder/preprocessor_config.json` 配置如下：

```json
{
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_processor_type": "Qwen2VLImageProcessor",
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "merge_size": 2,
  "patch_size": 16,
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "longest_edge": 16777216,
    "shortest_edge": 65536
  },
  "temporal_patch_size": 2,
  "processor_class": "Qwen3VLProcessor"
}
```

虽然本文只涉及图片生成，当前 Qwen3-VL Processor 初始化仍要求
`text_encoder/video_preprocessor_config.json` 存在，其配置如下：

```json
{
  "processor_class": "Qwen3VLProcessor",
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "do_sample_frames": true,
  "fps": 2,
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "max_frames": 768,
  "merge_size": 2,
  "min_frames": 4,
  "patch_size": 16,
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "return_metadata": false,
  "size": {
    "longest_edge": 25165824,
    "shortest_edge": 4096
  },
  "temporal_patch_size": 2,
  "video_processor_type": "Qwen3VLVideoProcessor"
}
```

## 4. 启动图片编辑服务

下面给出一个单机四 die 的已验证配置：

- `tp_size=1`
- `cfg_size=2`
- `sp_size=2`
- `vae_size=4`
- `text_encoder_tp_size=4`
- 默认关闭 DiT Cache

### 4.1 配置环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:${LD_LIBRARY_PATH}
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_IF_BASE_PORT=41465
export INF_NAN_MODE_ENABLE=0
export INF_NAN_MODE_FORCE_DISABLE=1
```

### 4.2 单机四 die 启动示例

以下示例使用逻辑设备 `0,1,2,3`。启动前请通过 `npu-smi info` 确认这些设备
拥有足够的空闲 HBM，并根据实际环境调整 `ASCEND_RT_VISIBLE_DEVICES`。

```bash
XLLM_PATH=./build/xllm/core/server/xllm
MODEL_PATH=/export/home/models/joyai-image-edit-plus
MODEL_ID=Joy_Image_Edit_2509
MASTER_NODE_ADDR=127.0.0.1:24581
START_PORT=18284
LOG_DIR=log/joy_image_edit_plus
NNODES=4

mkdir -p "${LOG_DIR}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

for ((rank = 0; rank < NNODES; rank++)); do
  port=$((START_PORT + rank))

  nohup setsid "${XLLM_PATH}" \
    --model="${MODEL_PATH}" \
    --model_id="${MODEL_ID}" \
    --backend=dit \
    --npu_kernel_backend=TORCH \
    --master_node_addr="${MASTER_NODE_ADDR}" \
    --nnodes="${NNODES}" \
    --node_rank="${rank}" \
    --port="${port}" \
    --communication_backend=hccl \
    --tp_size=1 \
    --cfg_size=2 \
    --sp_size=2 \
    --vae_size=4 \
    --text_encoder_tp_size=4 \
    --max_memory_utilization=0.8 \
    --dit_cache_policy=None \
    --dit_sp_communication_overlap=false \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --enable_shm=true \
    --use_contiguous_input_buffer=false \
    >"${LOG_DIR}/node_${rank}.log" 2>&1 < /dev/null &
  echo "$!" >"${LOG_DIR}/node_${rank}.pid"
done
```

当 rank 0 日志出现 `Application startup complete` 后，可通过健康检查确认服务状态：

```bash
curl http://127.0.0.1:18284/health
```

API 只需要访问 rank 0 的端口，其他端口用于分布式服务进程。

## 5. 发送图片编辑请求

请求接口为：

```text
POST /v1/image/generation
```

`input.images` 接收不带 Data URL 前缀的 Base64 图片字符串。下面的 Python
示例输入两张参考图片，将返回的第一张结果保存为 `output.png`：

```python
import base64
from pathlib import Path

import requests


def encode_image(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


payload = {
    "model": "Joy_Image_Edit_2509",
    "input": {
        "prompt": "将第一张图中的冲锋衣穿到第二张图片中的模特身上。",
        "negative_prompt": "low quality, blurry, deformed",
        "images": [
            encode_image("input_0.png"),
            encode_image("input_1.png"),
        ],
    },
    "parameters": {
        "size": "1024*1024",
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "true_cfg_scale": 4.0,
        "seed": 42,
        "max_sequence_length": 4096,
    },
    "user": "joy-image-edit-plus-example",
}

session = requests.Session()
session.trust_env = False
response = session.post(
    "http://127.0.0.1:18284/v1/image/generation",
    json=payload,
    timeout=1200,
)
response.raise_for_status()

result = response.json()
image_base64 = result["output"]["results"][0]["image"]
Path("output.png").write_bytes(base64.b64decode(image_base64))
print("图片已保存到 output.png")
```

`parameters` 中常用的请求参数如下：

| 参数 | 说明 | 建议值 |
| ---- | ---- | ------ |
| `size` | 输出图片尺寸，格式为 `宽*高` | `1024*1024` |
| `num_inference_steps` | 去噪步数，值越大通常耗时越长 | `30` |
| `true_cfg_scale` | 正负提示词 CFG 强度 | `4.0` |
| `guidance_scale` | 基础 guidance 参数 | `1.0` |
| `seed` | 随机种子，用于复现实验 | 任意非负整数 |
| `max_sequence_length` | Qwen3-VL 最大文本序列长度 | `4096` |

如果需要完全复现实验输入，也可以在 `input.latent` 中传入固定 latent Tensor；
普通在线图片编辑请求通常不需要设置该字段。

## 6. 并行参数说明

| 参数 | 说明 | 推荐配置 |
| ---- | ---- | -------- |
| `--cfg_size` | Classifier-Free Guidance 并行度 | 使用负向提示词时可设为 `2` |
| `--sp_size` | Transformer Sequence Parallel 并行度 | 四 die 示例设为 `2` |
| `--vae_size` | VAE 空间并行度 | 四 die 示例设为 `4` |
| `--text_encoder_tp_size` | 内置 Qwen3-VL text encoder 的 TP 并行度 | 四 die 示例设为 `4` |
| `--tp_size` | DiT Tensor Parallel 并行度 | 当前示例保持 `1` |
| `--dit_cache_policy` | DiT Cache 策略 | 默认建议 `None`；需要性能优化时再评估 Cache 精度 |
| `--dit_sp_communication_overlap` | SP 通信计算重叠 | 基础部署建议 `false` |

DiT 的分布式进程数满足：

```text
NNODES = tp_size * cfg_size * sp_size
```

`vae_size` 和 `text_encoder_tp_size` 使用同一组服务进程建立各自的并行组，
不额外增加 `NNODES`。

| `tp_size` | `cfg_size` | `sp_size` | `vae_size` | `text_encoder_tp_size` | `NNODES` | 说明 |
| --------- | ---------- | --------- | ---------- | ---------------------- | -------- | ---- |
| `1` | `1` | `1` | `1` | `1` | `1` | 单 die 基础配置 |
| `1` | `2` | `1` | `2` | `2` | `2` | 两 die CFG 并行 |
| `1` | `2` | `2` | `4` | `4` | `4` | 四 die CFG + SP 推荐配置 |
