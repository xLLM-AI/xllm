---
title: "Supported Models"
description: "Find an xLLM model by task and accelerator, compare documented hardware support, and open deployment guides."
sidebar:
  order: 20
---
<p class="catalog-intro">Find the right model for your workload. Compare hardware support across text, vision, reranking, generation, and recommendation, then open a deployment guide to get started.</p>
<div class="catalog-links">
<a href="/en/getting_started/quick_start/">Quick start →</a>
<a href="/en/hardware/overview/">Hardware setup →</a>
<a href="/en/getting_started/launch_xllm/">Launch a service →</a>
</div>

<nav class="catalog-jumps" aria-label="Model categories">
<a href="#llm"><strong>LLM</strong><span>Text generation</span></a>
<a href="#vlm"><strong>VLM</strong><span>Vision-language</span></a>
<a href="#rerank"><strong>Rerank</strong><span>Reranking</span></a>
<a href="#dit"><strong>DiT</strong><span>Image & video</span></a>
<a href="#rec"><strong>Rec</strong><span>Recommendation</span></a>
</nav>

<div data-catalog-filters></div>

<div class="catalog-legend">
<span><b data-support="supported">✓</b> Supported</span>
<span><b data-support="unsupported">×</b> Not supported</span>
<span><b data-support="unknown">—</b> Not documented</span>
</div>

<section data-model-category="llm">

## LLM

Language models for chat, reasoning, and text generation.

<div class="catalog-table" tabindex="0" role="region" aria-label="Text generation">

| Model / family | Ascend NPU | Cambricon MLU | Hygon DCU | Deployment guide |
| :--- | :---: | :---: | :---: | :--- |
| DeepSeek-V3 / R1 / V3.1 | ✅ | ✅ | ❌ | Not yet available |
| DeepSeek-V3.2 | ✅ | ✅ | ❌ | Not yet available |
| DeepSeek-V4 | ✅ | ✅ | — | [Guide ↗](/en/cookbook/autoregressive_models/deepseek/deepseek_v4/) |
| Qwen2 / Qwen2.5 / QwQ | ✅ | ✅ | ✅ | Not yet available |
| Qwen3 | ✅ | ✅ | ✅ | Not yet available |
| Qwen3 MoE | ✅ | ✅ | ✅ | Not yet available |
| Qwen3.5 | ✅ | ✅ | — | [Guide ↗](/en/cookbook/autoregressive_models/qwen/qwen3_5/) |
| Kimi-K2 | ✅ | ❌ | ❌ | Not yet available |
| Kimi-K2.5 / Kimi-K2.6 | ✅ | — | — | [Guide ↗](/en/cookbook/autoregressive_models/kimi/kimi2_5/) |
| Llama2 / Llama3 | ✅ | ❌ | ❌ | Not yet available |
| GLM-4.5 | ✅ | ❌ | ❌ | Not yet available |
| GLM-4.6 | ✅ | ❌ | ❌ | Not yet available |
| GLM-4.7 | ✅ | ❌ | ❌ | Not yet available |
| GLM-5 | ✅ | ✅ | ❌ | [Guide ↗](/en/cookbook/autoregressive_models/glm/glm_5/) |
| GLM-5.1 / GLM-5.2 / GLM-5.3 | ✅ | ✅ | — | [Guide ↗](/en/cookbook/autoregressive_models/glm/glm_5/) |
| GLM-5.3-Flash | ✅ | — | — | [Preview guide ↗](/en/cookbook/autoregressive_models/glm/glm_5_3_flash/) |
| MiniMax-M2.7 | ✅ | — | — | [Preview guide ↗](/en/cookbook/autoregressive_models/minmax/minmax_m2_7/) |

</div>

</section>

<section data-model-category="vlm">

## VLM

Vision-language models for understanding images and text.

<div class="catalog-table" tabindex="0" role="region" aria-label="Vision-language">

| Model / family | Ascend NPU | Cambricon MLU | Hygon DCU | Deployment guide |
| :--- | :---: | :---: | :---: | :--- |
| MiniCPM-V | ✅ | ❌ | ❌ | Not yet available |
| MiMo-VL | ✅ | ❌ | ❌ | Not yet available |
| Qwen2.5-VL | ✅ | ✅ | ✅ | Not yet available |
| Qwen3-VL | ✅ | ✅ | ✅ | Not yet available |
| GLM-4.6V | ✅ | ❌ | ❌ | Not yet available |
| GLM-5.3-Flash-VL | ✅ | — | — | [Preview guide ↗](/en/cookbook/autoregressive_models/glm/glm_5_3_flash/) |
| VLM-R1 | ✅ | ❌ | ❌ | Not yet available |

</div>

</section>

<section data-model-category="rerank">

## Rerank

Models for scoring and reordering retrieved documents.

<div class="catalog-table" tabindex="0" role="region" aria-label="Reranking">

| Model / family | Ascend NPU | Cambricon MLU | Hygon DCU | Deployment guide |
| :--- | :---: | :---: | :---: | :--- |
| Qwen3-Reranker | ✅ | ❌ | ❌ | Not yet available |

</div>

</section>

<section data-model-category="dit">

## DiT

Diffusion models for image and video generation.

<div class="catalog-table" tabindex="0" role="region" aria-label="Image & video">

| Model / family | Ascend NPU | Cambricon MLU | Hygon DCU | Deployment guide |
| :--- | :---: | :---: | :---: | :--- |
| Flux | ✅ | ❌ | ❌ | Not yet available |
| Flux2 | ✅ | — | — | [Guide ↗](/en/cookbook/diffusion_models/flux/flux2/) |
| Wan2.2 | ✅ | — | — | [Guide ↗](/en/cookbook/diffusion_models/wan/wan2_2/) |

</div>

</section>

<section data-model-category="rec">

## Rec

Models for generative recommendation workloads.

<div class="catalog-table" tabindex="0" role="region" aria-label="Recommendation">

| Model / family | Ascend NPU | Cambricon MLU | Hygon DCU | Deployment guide |
| :--- | :---: | :---: | :---: | :--- |
| OneRec | ✅ | ❌ | ❌ | Not yet available |
| Qwen2 | ✅ | ❌ | ❌ | Not yet available |
| Qwen2.5 | ✅ | ❌ | ❌ | Not yet available |
| Qwen3 | ✅ | ❌ | ❌ | Not yet available |

</div>

</section>
