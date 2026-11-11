---
title: "DeepSeek-OCR Deployment"
description: "Serving DeepSeek-OCR with xLLM on Ascend devices"
---

+ Model card: https://huggingface.co/deepseek-ai/DeepSeek-OCR (MIT)

+ Weights are also mirrored on ModelScope: `deepseek-ai/DeepSeek-OCR`

DeepSeek-OCR is a 3.3B document-OCR vision-language model: a SAM-ViT-B
window-attention tower feeds a CLIP-L refine tower (the CLIP patch conv is
bypassed; the SAM grid is consumed as patch embeddings), and a linear
projector maps the concatenated features into a 12-layer DeepSeek-V2-style
decoder (plain MHA, dense FFN at layer 0, MoE at layers 1-11 with 64 routed
experts top-6). This cookbook covers serving it through the Python model
executor on a single Ascend 910 card.

### Note

Single card only. The MoE implementation rejects `tp_size/dp_size > 1` at
startup with an explicit error; see the multi-card discussion on the
supporting PR. With 6.7 GB bf16 weights on a 64 GB card, a single instance
already saturates the card for this workload.

## 1. Build

```bash
python setup.py install --device npu
```

## 2. Serve

```bash
xllm serve \
  --model=/path/to/DeepSeek-OCR \
  --model_id=DeepSeek-OCR \
  --backend=vlm \
  --model_impl=python \
  --npu_kernel_backend=TORCH \
  --port=8011 \
  --nnodes=1 \
  --block_size=128 \
  --communication_backend=hccl \
  --max_memory_utilization=0.86 \
  --enable_prefix_cache=false \
  --enable_chunked_prefill=false \
  --enable_schedule_overlap=false \
  --enable_shm=false \
  --limit_image_per_prompt=4
```

`--enable_prefix_cache=false --enable_chunked_prefill=false` are required
for python-executor VLMs (image embeddings are merged over the full
flattened batch).

## 3. Request

The prompt carries a `<image>` placeholder per image; images ride along as
base64 data URLs in OpenAI content-parts format.

```bash
B64=$(base64 -w0 /path/to/document.jpg)
curl -s http://127.0.0.1:8011/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"DeepSeek-OCR\",\"messages\":[{\"role\":\"user\",\"content\":[
    {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,$B64\"}},
    {\"type\":\"text\",\"text\":\"<image>\nFree OCR. \"}]}],
    \"max_tokens\":300,\"temperature\":0}"
```

Common prompts:

- `<image>\nFree OCR. ` — free-form transcription
- `<image>\n<|grounding|>Convert the document to markdown. ` — markdown with
  bounding-box annotations (`<|ref|>label<|/ref|><|det|>[x1,y1,x2,y2]<|/det|>`,
  coordinates normalized to 0-999)

## 4. Preprocessing recipe

Serving uses the Gundam recipe (HF `infer()` defaults): the image is padded
to a 1024x1024 global view and, when a side exceeds 640 px, additionally
tiled into 640x640 crops by aspect ratio. Both views go through the towers
and their token grids are spliced into the prompt. A 2x2-tiled page expands
to 693 `<image>` tokens (273 for the global view + 420 for the crops grid);
grid geometry is checkpoint-fixed (patch 16, 4x downsample).

## 5. Reference numbers

Single Ascend 910 (64 GB), idle NPU, 699-token Gundam prompts, batch 1/4:

| Scenario | Result |
| --- | --- |
| Text-only decode | 110.9 tok/s |
| OCR streaming TTFT (incl. vision encode) | 0.25 s |
| OCR single request (699 -> 300 tok) | 104.0 out-tok/s |
| OCR x4 concurrent | 305 tok/s aggregate |

## 6. Troubleshooting

- **`NotImplementedError: deepseek_ocr MoE currently supports
  dp=moe_tp=ep=1 only`** — you launched with `--tp_size/--dp_size/--ep_size
  > 1`; the model is single-card for now.
- **`architectures 'X' is not supported for model_type deepseek_vl_v2`** —
  you pointed `--model` at an original DeepSeek-VL2 checkpoint; only
  `DeepseekOCRForCausalLM` (DeepSeek-OCR) is implemented.
- **Long tables repeat themselves** — the release expects an n-gram repeat
  suppressor on the sampling side (HF `no_repeat_ngram_size=20`, vLLM
  `NGramPerReqLogitsProcessor`); porting it to xLLM's logits processors is
  tracked follow-up work.
