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

按照 xLLM 的 [Ascend NPU 环境指南](/zh/hardware/ascend_npu/)
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
其中声明的组件目录均需保留。准备好 Hugging Face 的 `hf` CLI 后，
将组件下载到自己的权重目录：

```bash
export MODEL_PATH=/path/to/MiniMax-H3

hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --local-dir "$MODEL_PATH" \
  --include model_index.json --include 'transformer/*' \
  --include 'transformer_ref/*' --include 'vae/*' --include 'audio_vae/*' \
  --include 'scheduler/*' --include 'audio_scheduler/*' --include 'processor/*' \
  --include 'text_encoder/*' --include 'tokenizer/*'
```

## 文本条件

当前 H3 管线要求调用方提供与模型配套的文本编码器生成的文本条件，
不会仅根据 `input.prompt` 自动执行文本编码。xLLM 当前不提供 H3 文本编码的
预处理命令；以下请求从已有编码器输出开始，不能直接仅用原始文本运行。
先取得配套 H3 编码器生成的特征和 token 标签，再将包含以下字段的 PyTorch
字典保存为 `conditioning.pt`。提示词、特征和标签必须来自同一次编码：

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
export MODEL_PATH=/path/to/MiniMax-H3
export XLLM_BINARY=/path/to/xllm
export OMP_NUM_THREADS=8
export HCCL_BUFFSIZE=256

python -m xllm.launch_server serve --binary-path="$XLLM_BINARY" \
  --nnodes=4 --backend=dit --model="$MODEL_PATH" --model_id=minimax-h3 \
  --tp_size=2 --sp_size=2 --cfg_size=1 --vae_size=4 \
  --text_encoder_tp_size=1 --dit_h3_task_type=fl2va --dit_cache_policy=None \
  --dit_laser_attention_enabled=false --dit_sparse_attention_enabled=false \
  --master_node_addr=127.0.0.1:19885 --host=127.0.0.1 --port=8000 \
  --max_body_size=2147483648 --logtostderr=true
```

八卡时，设置 `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`，并修改以下参数：

| 配置 | `--nnodes` | `--tp_size` | `--sp_size` | `--vae_size` |
| --- | --- | --- | --- | --- |
| 四卡 | `4` | `2` | `2` | `4` |
| 八卡 | `8` | `2` | `4` | `8` |

`--vae_size` 设置独立于 TP/SP 的 VAE 进程组大小，不计入
`TP * SP * CFG * DP` 的乘积；其值必须为正数，且能整除 `--nnodes`。
上述配置将视频空间切片分配到所有 NPU 上解码，再聚合、拼接输出；
该参数不会并行音频 VAE 解码。
VAE 保持 FP32，每张参与的 NPU 都会加载完整的视频 VAE 权重，并保存聚合后的切片；
启用前需为这些权重和解码中间张量留出显存。

设置 `--vae_size=1` 可使用本地 VAE 解码，无需更改 TP/SP 配置。
未触发空间切片或切片数小于 VAE 组大小时，各参与 rank 使用本地解码路径，
不会因增加 `--vae_size` 获得切片并行收益。

`--max_body_size` 为生成结果的内部传输预留空间，不能按压缩后 MP4 的大小设置。
标准权重有 56 个注意力头、96 个视频 patch 通道和 32 个音频通道；并行配置需满足
`heads % (TP * SP) == 0`、`video_patch_dim % TP == 0` 和
`audio_in_channels % TP == 0`。

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

## Transformer 缓存

H3 默认使用完整计算。要开启 TaylorSeer 近似缓存，必须显式指定
`--dit_cache_policy=TaylorSeer`，或在 JSON 配置中设置
`"dit_cache_policy": "TaylorSeer"`。对于 50 步请求，可将服务启动命令中的缓存参数改为：

```bash
--dit_cache_policy=TaylorSeer \
--dit_cache_warmup_steps=45 \
--dit_cache_skip_interval_steps=2 \
--dit_cache_n_derivatives=1
```

计算间隔必须大于零；设为 `1` 时每个去噪步都完整计算。预热步数和导数阶数必须
非负。缓存历史按请求和 SP 分片隔离。缓存近似的是 video/audio/text 打包序列的
Transformer block 输出；依赖当前时间步的输出归一化、媒体投影、采样器更新和
VAE 解码仍正常执行。近似缓存会改变数值结果，需要针对实际任务验证生成质量。
使用 `--dit_cache_policy=None` 可恢复完整计算。

要用 FBCache 替换 TaylorSeer 或完整计算，请将服务启动命令中的缓存参数
替换为以下内容。这组设置用于 50 步请求；文本条件和 HTTP 请求格式保持一致：

```bash
--dit_cache_policy=FBCache \
--dit_cache_warmup_steps=45 \
--dit_cache_residual_diff_threshold=0.09
```

预热步数和残差阈值必须非负。阈值设为 `0` 时不复用缓存；预热步数不小于请求的
去噪步数时，每步也均完整计算。近似复用会改变数值结果，需针对实际任务同时检查
视频和音频质量来选择参数。

FBCache 不能与滚动加载权重同时使用。TP=1 时必须显式设置
`--enable_rolling_load=false`，此时单个设备须能容纳完整常驻权重及推理工作区；
关闭滚动加载并不保证 TP=1 能够装下模型。显存不足时，请使用上面的多卡布局。
H3 仅支持 `None`、`TaylorSeer` 和 `FBCache`，不支持 `FBCacheTaylorSeer`。
