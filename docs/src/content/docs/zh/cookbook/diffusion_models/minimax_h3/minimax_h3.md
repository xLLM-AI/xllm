---
title: "MiniMax-H3 部署与使用"
sidebar:
  order: 1
---

本文介绍如何在 Ascend NPU 上使用 xLLM 部署 MiniMax-H3，并通过
`/v1/video/generation` 接口生成带音频的视频。当前已验证 BF16 推理、
四卡 TP2/SP2 和八卡 TP2/SP4；使用 `fl2va` 管线，不传图像条件时执行
文生音视频（T2VA）。

## 环境准备

按照 xLLM [Ascend NPU 环境指南](/zh/hardware/ascend_npu/)
准备开发环境，并在 xLLM 仓库根目录构建：

```bash
git submodule update --init --recursive
python setup.py build --device npu
```

运行服务前加载 Ascend 环境及依赖库路径，并安装 FFmpeg，确保
`ffmpeg` 和 `ffprobe` 在 `PATH` 中。启用 etcd 时，还需准备
`third_party/Mooncake/mooncake-common/etcd/go.mod` 要求的 Go 工具链。

## 权重准备

使用 `MiniMaxAI/MiniMax-H3` 根目录的 Diffusers 组件格式。
`model_index.json` 的 `_class_name` 应为 `MiniMaxH3ModularPipeline`，
其中声明的组件目录均需保留。下面下载已验证版本的组件：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/models/.cache/huggingface
export MODEL_PATH=/data/models/MiniMax-H3

uvx --from huggingface_hub==1.33.0 hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --local-dir "$MODEL_PATH" \
  --include model_index.json --include 'transformer/*' \
  --include 'transformer_ref/*' --include 'vae/*' --include 'audio_vae/*' \
  --include 'scheduler/*' --include 'audio_scheduler/*' --include 'processor/*' \
  --include 'text_encoder/*' --include 'tokenizer/*'
```

## 文本条件

当前 H3 管线要求调用方提供与模型配套的文本编码器生成的文本条件，
不会仅根据 `input.prompt` 自动执行文本编码。准备一个 `conditioning.pt`
文件，包含以下字段；文本条件必须与请求提示词一致：

| 字段 | 类型与形状 | 说明 |
| --- | --- | --- |
| `prompt` | 字符串 | 生成该文本条件时使用的提示词 |
| `prompt_embed` | BF16 Tensor，`[1, L, 5120]` | H3 文本特征 |
| `prompt_token_tags` | INT64 Tensor，`[L]` | 编码器输出的 token 标签，取值为 `0` 或 `1` |

HTTP 请求中的 embedding 形状为 `[L, 5120]`，服务端添加 batch 维度。
token 标签放在该张量的 `parameters.prompt_token_tags.int64_tensor_param` 中。
不能用随机张量、其他模型的 embedding 或不匹配的标签代替。

## 启动服务

在四张可用 NPU 上启动服务。`XLLM_BINARY` 指向构建得到的 xLLM 可执行文件：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MODEL_PATH=/data/models/MiniMax-H3
export XLLM_BINARY=/path/to/xllm
export OMP_NUM_THREADS=8
export HCCL_BUFFSIZE=256

python -m xllm.launch_server serve --binary-path="$XLLM_BINARY" \
  --nnodes=4 --backend=dit --model="$MODEL_PATH" --model_id=minimax-h3 \
  --tp_size=2 --sp_size=2 --cfg_size=1 --vae_size=1 \
  --text_encoder_tp_size=1 --dit_h3_task_type=fl2va --dit_cache_policy=None \
  --dit_laser_attention_enabled=false --dit_sparse_attention_enabled=false \
  --master_node_addr=127.0.0.1:19885 --host=127.0.0.1 --port=8000 \
  --max_body_size=2147483648 --logtostderr=true
```

八卡时，设置 `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`，并修改以下参数：

| 配置 | `--nnodes` | `--tp_size` | `--sp_size` |
| --- | --- | --- | --- |
| 四卡 | `4` | `2` | `2` |
| 八卡 | `8` | `2` | `4` |

`--max_body_size` 为生成结果的内部传输预留空间，不能按压缩后 MP4 的大小设置。
标准权重有 56 个注意力头、96 个视频 patch 通道和 32 个音频通道；并行配置需满足
`heads % (TP * SP) == 0`、`video_patch_dim % TP == 0` 和
`audio_in_channels % TP == 0`。

## Laser Attention

在 Ascend NPU 上启用 Laser 时，将上述服务启动参数改为
`--dit_laser_attention_enabled=true`，默认值为 `false`。继续使用本文说明的
checkpoint、文本条件、请求字段及并行配置；启用 Laser 不会运行文本编码器。

需要一起构建 xLLM 和匹配的 `xllm_ops` 子模块。更新原生算子时使用新的
构建目录，避免复用旧的内核生成缓存。运行环境沿用上述 Ascend 环境。

H3 对有效长度至少为 2,048、head dimension 不超过 128 的 self-attention
选择 Laser，短序列使用现有 dense attention 算子。H3 会排除 SP padding，
Laser 自行处理序列长度与 head dimension 的对齐；用户无需额外补齐输入或
缩放输入数值范围。

该选项提供 dense、非因果注意力，不增加外部 attention mask、dropout 或
GQA 支持。下述模型和请求限制仍然适用。

## 生成音视频

在包含 `conditioning.pt` 的目录执行下面的请求。客户端需要安装 PyTorch，
用于读取文本条件；HTTP 和 Base64 处理使用 Python 标准库。

```bash
python - <<'PY'
import base64
import json
import urllib.request
from pathlib import Path

import torch

conditioning = torch.load("conditioning.pt", map_location="cpu", weights_only=True)
embedding = conditioning["prompt_embed"].squeeze(0).float().contiguous()
payload = {
    "model": "minimax-h3",
    "input": {
        "prompt": conditioning["prompt"],
        "prompt_embed": {
            "name": "prompt_embed",
            "datatype": "FP32",
            "shape": list(embedding.shape),
            "contents": {"fp32_contents": embedding.flatten().tolist()},
            "parameters": {"prompt_token_tags": {"int64_tensor_param": conditioning["prompt_token_tags"].tolist()}},
        },
    },
    "parameters": {
        "size": "1344*768",
        "num_inference_steps": 50,
        "seed": 42,
        "num_frames": 124,
        "fps": 24,
        "sampling_rate": 32000,
        "flow_shift": 12.0,
        "guidance_scale": 1.0,
        "num_videos_per_prompt": 1,
        "output_type": "base64",
    },
}
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/video/generation",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=3600) as response:
    result = json.load(response)
video = result["output"]["results"][0]["video"]["base64"]
Path("output.mp4").write_bytes(base64.b64decode(video, validate=True))
PY
```

结果保存在 `output.mp4`，包含视频和 32 kHz 双声道音频。可用以下命令检查媒体信息：

```bash
ffprobe -v error -show_entries \
  stream=codec_type,width,height,nb_frames,r_frame_rate,sample_rate,channels \
  -of json output.mp4
```

当前服务仅支持 batch 1 和 `num_videos_per_prompt=1`。分辨率和帧数会按模型要求
对齐，对齐后时长需在 5～15 秒之间；上述请求生成 1344×768、124 帧、24 FPS 的音视频。
其他任务、权重格式、精度和并行布局需单独验证。
