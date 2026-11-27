---
title: "Wan2.1"
description: "Wan2.1-I2V-14B 的 Python 和 libtorch 组图"
---

## 模型和依赖

使用 `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` 或
`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` 权重。
模型根目录需要 `model_index.json`，以及 `transformer`、`text_encoder`、
`tokenizer`、`image_encoder`、`image_processor`、`vae`、`scheduler` 目录。
原始 Wan 仓库的 `.pth` 权重不能直接用于此路径。

运行环境需要 PyTorch、torch_npu、Diffusers 0.43.0、Transformers、
`ftfy` 和 `sentencepiece`。文本编码使用 UMT5 SentencePiece tokenizer，
图像编码使用 CLIP，去噪调度使用 checkpoint 的 UniPC flow-matching 配置。

## 启动配置

以下是四卡 TP2/SP2 的启动配置，需在模型下载完成后使用：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MODEL_PATH=/path/to/Wan2.1-I2V-14B-720P-Diffusers
export XLLM_BINARY=/path/to/xllm
export OMP_NUM_THREADS=8
export HCCL_BUFFSIZE=256

python -m xllm.launch_server serve --binary-path="$XLLM_BINARY" \
  --nnodes=4 --backend=dit --model="$MODEL_PATH" --model_id=wan21-i2v \
  --model_impl=python --tp_size=2 --sp_size=2 --cfg_size=1 --vae_size=1 \
  --text_encoder_tp_size=1 --max_requests_per_batch=1 \
  --dit_cache_policy=None --enable_graph=false --enable_rolling_load=false \
  --dit_laser_attention_enabled=false --dit_sparse_attention_enabled=false \
  --master_node_addr=127.0.0.1:19885 --host=127.0.0.1 --port=8000 \
  --max_body_size=2147483648 --logtostderr=true
```

设置 `--model_impl=libtorch` 可使用 libtorch 组图。Python 组图使用 dense eager
执行。TP 切分注意力和 FFN 权重，SP 切分
视频 token；`TP * SP` 必须整除 40 个注意力头，TP 必须整除 FFN 宽度。
CFG、DP、VAE 和文本编码器 TP 均设为 1。VAE 使用 FP32 完整空间编码，
去噪后仅 driver 执行解码。

## 请求约定

`POST /v1/video/generation` 使用现有视频生成协议：

- `input.prompt`、可选的 `input.negative_prompt`：在服务内编码文本。
- `input.media_sources`：恰好一个 `name="image"`、`modality="image"` 的
  Base64 或 binary 图像来源。
- 可选的 `input.prompt_embed` 和 `input.negative_prompt_embed`：UMT5
  浮点 `[512, 4096]` 特征，包含补零后的 512 个 token；服务端添加 batch 维度。
- 推荐设置 `num_frames=81`、`fps=16`、`num_inference_steps=50`、
  `guidance_scale=5.0`、`max_sequence_length=512`、`num_videos_per_prompt=1`、
  `seed=42`。720P 使用 `size="1280*720"`、`flow_shift=5.0`，480P 使用
  `size="832*480"`、`flow_shift=3.0`。

宽高必须为正的 16 的倍数，帧数按 `floor(num_frames / 4) * 4 + 1`
对齐。`flow_shift` 使用请求值，未指定时遵循 xLLM 的默认值 1.0。
输出沿用原有 MP4 编码和 Base64/binary 返回协议；此模型不生成音频。
