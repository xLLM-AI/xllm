---
title: "DeepSeek-OCR Deployment"
description: "在 Ascend 设备上使用 xLLM 服务 DeepSeek-OCR"
---

+ 模型主页：https://huggingface.co/deepseek-ai/DeepSeek-OCR（MIT 协议）

+ 权重同步镜像于 ModelScope：`deepseek-ai/DeepSeek-OCR`

DeepSeek-OCR 是一个 3.3B 参数的文档 OCR 视觉语言模型：SAM-ViT-B 窗口注意力塔
输出特征图后由 CLIP-L 精炼塔消费（CLIP 的 patch 卷积被旁路，直接以 SAM 网格作为
patch embedding），拼接特征经线性投影层进入 12 层 DeepSeek-V2 风理解码器
（标准 MHA，第 0 层稠密 FFN，第 1-11 层 MoE：64 路由专家 top-6）。本指南覆盖在
单张 Ascend 910 上通过 Python model executor 服务该模型。

### 说明

仅支持单卡。MoE 实现在启动时对 `tp_size/dp_size > 1` 显式报错；多卡讨论见
对应 PR。6.7 GB 的 bf16 权重对 64 GB 显存的卡而言余量充足，单实例即可打满该
负载。

## 1. 构建

```bash
python setup.py install --device npu
```

## 2. 启动服务

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

`--enable_prefix_cache=false --enable_chunked_prefill=false` 对
python-executor VLM 是必需的（图像 embedding 在完整扁平批次上合并）。

## 3. 发起请求

prompt 中每张图占一个 `<image>` 占位符；图像以 base64 data URL 的
OpenAI content-parts 形式传入。

```bash
B64=$(base64 -w0 /path/to/document.jpg)
curl -s http://127.0.0.1:8011/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"DeepSeek-OCR\",\"messages\":[{\"role\":\"user\",\"content\":[
    {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,$B64\"}},
    {\"type\":\"text\",\"text\":\"<image>\nFree OCR. \"}]}],
    \"max_tokens\":300,\"temperature\":0}"
```

常用 prompt：

- `<image>\nFree OCR. ` —— 自由转写
- `<image>\n<|grounding|>Convert the document to markdown. ` —— 输出带
  坐标标注的 markdown（`<|ref|>标签<|/ref|><|det|>[x1,y1,x2,y2]<|/det|>`，
  坐标归一化到 0-999）

## 4. 预处理配方

服务使用 Gundam 配方（HF `infer()` 默认值）：图像先 pad 成 1024x1024 全局
视图；任一边超过 640 px 时再按宽高比切成 640x640 的局部视图。两种视图分别过塔，
token 网格拼接进 prompt。2x2 切图的整页会展开为 693 个 `<image>` token
（全局视图 273 + 切图网格 420）；网格几何由 checkpoint 固定
（patch 16、4 倍下采样）。

## 5. 参考性能

单张 Ascend 910（64 GB），空闲 NPU，699-token Gundam prompt，并发 1/4：

| 场景 | 结果 |
| --- | --- |
| 纯文本解码 | 110.9 tok/s |
| OCR 流式 TTFT（含视觉编码） | 0.25 s |
| OCR 单请求（699 -> 300 tok） | 104.0 out-tok/s |
| OCR x4 并发 | 305 tok/s 聚合 |

## 6. 常见问题

- **`NotImplementedError: deepseek_ocr MoE currently supports
  dp=moe_tp=ep=1 only`** —— 使用了 `--tp_size/--dp_size/--ep_size > 1`
  启动；该模型目前仅支持单卡。
- **`architectures 'X' is not supported for model_type deepseek_vl_v2`** ——
  `--model` 指向了原版 DeepSeek-VL2 权重；仅实现
  `DeepseekOCRForCausalLM`（DeepSeek-OCR）。
- **长表格内容重复** —— 该模型依赖采样侧的 n-gram 重复抑制（HF 侧为
  `no_repeat_ngram_size=20`，vLLM 侧为 `NGramPerReqLogitsProcessor`）；
  移植到 xLLM logits processor 属于后续跟进项。
