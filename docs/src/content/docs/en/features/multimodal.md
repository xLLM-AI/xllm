---
title: "Multimodal Support"
sidebar:
  order: 60
---
This document introduces the current multimodal support in the xLLM inference engine, including supported models, modality types, and offline and online interfaces.

## Supported Models
- Qwen2.5-VL: including 7B/32B/72B.
- Qwen3-VL: including 2B/4B/8B/32B.
- Qwen3-VL-MoE: including A3B/A22B.
- MiniCPM-V-2_6: 7B.

## Modality Types
- Images: supports single-image and multi-image inputs, image + prompt combinations, and text-only prompts.

:::caution[Notes]
- The multimodal backend does not currently support prefix cache or chunked prefill. Support is in progress.
- xLLM now renders ChatTemplate uniformly based on JinJa. When deploying MiniCPM-V-2_6, the model directory must provide a ChatTemplate file.
- Image inputs support Base64 data and image URLs.
- Multimodal models currently mainly support the image modality. Video, audio, and other modalities are in progress.

:::
