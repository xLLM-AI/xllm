---
title: "Wan2.1"
description: "Python and libtorch DiT execution for Wan2.1-I2V-14B"
---

## Checkpoint and dependencies

Use `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` or
`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`. The Diffusers layout includes
`model_index.json` and the `transformer`, `text_encoder`,
`tokenizer`, `image_encoder`, `image_processor`, `vae`, and `scheduler`
directories. Original Wan `.pth` checkpoints cannot be loaded directly.

Install PyTorch, torch_npu, Diffusers 0.43.0, Transformers, `ftfy`, and
`sentencepiece`. Text conditioning uses the UMT5 SentencePiece tokenizer,
image conditioning uses CLIP, and denoising uses the checkpoint's UniPC
flow-matching configuration.

## Server configuration

Use this four-device TP2/SP2 configuration once the checkpoint is available:

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

Set `--model_impl=libtorch` to use the libtorch pipeline. The Python pipeline
uses dense eager execution. TP shards attention
and FFN weights; SP shards video tokens. `TP * SP` must divide the 40
attention heads, and TP must divide the FFN width. Set CFG, DP, VAE, and text
encoder TP to 1. The VAE encodes the full spatial extent in FP32; only the
driver performs decoding after denoising.

## Request contract

`POST /v1/video/generation` uses the existing video generation protocol:

- `input.prompt` and optional `input.negative_prompt` are encoded in the server.
- `input.media_sources` contains exactly one Base64 or binary image source
  with `name="image"` and `modality="image"`.
- Optional `input.prompt_embed` and `input.negative_prompt_embed` accept
  floating point UMT5 features of shape `[512, 4096]`, including zero-padded
  text tokens. The server adds the batch dimension.
- Suggested parameters are `num_frames=81`, `fps=16`, `num_inference_steps=50`,
  `guidance_scale=5.0`, `max_sequence_length=512`, `num_videos_per_prompt=1`,
  and `seed=42`. For 720P, use `size="1280*720"` and `flow_shift=5.0`. For
  480P, use `size="832*480"` and `flow_shift=3.0`.

Height and width must be positive multiples of 16. Frames are aligned to
`floor(num_frames / 4) * 4 + 1`. The request controls `flow_shift`; omitting it
retains xLLM's default of 1.0. Results use the existing MP4 encoding and
Base64/binary response contract. This model does not generate audio.
