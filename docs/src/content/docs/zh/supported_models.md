---
title: "支持的模型"
description: "按任务和硬件查找 xLLM 模型，对比已记录的平台支持情况，并查看部署指南。"
sidebar:
  order: 20
---
<p class="catalog-intro">为你的任务选择合适的模型。对比文本生成、视觉语言、重排序、图像与视频生成及推荐模型的硬件支持情况，从部署指南开始使用。</p>
<div class="catalog-links">
<a href="/zh/getting_started/quick_start/">快速开始 →</a>
<a href="/zh/hardware/overview/">硬件环境配置 →</a>
<a href="/zh/getting_started/launch_xllm/">启动服务 →</a>
</div>

<nav class="catalog-jumps" aria-label="模型分类">
<a href="#llm"><strong>LLM</strong><span>文本生成</span></a>
<a href="#vlm"><strong>VLM</strong><span>视觉语言</span></a>
<a href="#rerank"><strong>Rerank</strong><span>重排序</span></a>
<a href="#dit"><strong>DiT</strong><span>图像与视频生成</span></a>
<a href="#rec"><strong>Rec</strong><span>生成式推荐</span></a>
</nav>

<div data-catalog-filters></div>

<div class="catalog-legend">
<span><b data-support="supported">✓</b> 支持</span>
<span><b data-support="unsupported">×</b> 未支持</span>
<span><b data-support="unknown">—</b> 未确认</span>
</div>

<section data-model-category="llm">

## LLM

用于对话、推理和文本生成的语言模型。

<div class="catalog-table" tabindex="0" role="region" aria-label="文本生成">

| 模型 / 系列 | Ascend NPU | Cambricon MLU | Hygon DCU | 部署指南 |
| :--- | :---: | :---: | :---: | :--- |
| DeepSeek-V3 / R1 / V3.1 | ✅ | ✅ | ❌ | 指南待补充 |
| DeepSeek-V3.2 | ✅ | ✅ | ❌ | 指南待补充 |
| DeepSeek-V4 | ✅ | ✅ | — | [部署指南 ↗](/zh/cookbook/autoregressive_models/deepseek/deepseek_v4/) |
| Qwen2 / Qwen2.5 / QwQ | ✅ | ✅ | ✅ | 指南待补充 |
| Qwen3 | ✅ | ✅ | ✅ | 指南待补充 |
| Qwen3 MoE | ✅ | ✅ | ✅ | 指南待补充 |
| Qwen3.5 | ✅ | ✅ | — | [部署指南 ↗](/zh/cookbook/autoregressive_models/qwen/qwen3_5/) |
| Kimi-K2 | ✅ | ❌ | ❌ | 指南待补充 |
| Kimi-K2.5 / Kimi-K2.6 | ✅ | — | — | [部署指南 ↗](/zh/cookbook/autoregressive_models/kimi/kimi2_5/) |
| Llama2 / Llama3 | ✅ | ❌ | ❌ | 指南待补充 |
| GLM-4.5 | ✅ | ❌ | ❌ | 指南待补充 |
| GLM-4.6 | ✅ | ❌ | ❌ | 指南待补充 |
| GLM-4.7 | ✅ | ❌ | ❌ | 指南待补充 |
| GLM-5 | ✅ | ✅ | ❌ | [部署指南 ↗](/zh/cookbook/autoregressive_models/glm/glm_5/) |
| GLM-5.1 / GLM-5.2 / GLM-5.3 | ✅ | ✅ | — | [部署指南 ↗](/zh/cookbook/autoregressive_models/glm/glm_5/) |
| GLM-5.3-Flash | ✅ | — | — | [预览版指南 ↗](/zh/cookbook/autoregressive_models/glm/glm_5_3_flash/) |
| MiniMax-M2.7 | ✅ | — | — | [预览版指南 ↗](/zh/cookbook/autoregressive_models/minmax/minmax_m2_7/) |

</div>

</section>

<section data-model-category="vlm">

## VLM

用于理解图像与文本的视觉语言模型。

<div class="catalog-table" tabindex="0" role="region" aria-label="视觉语言">

| 模型 / 系列 | Ascend NPU | Cambricon MLU | Hygon DCU | 部署指南 |
| :--- | :---: | :---: | :---: | :--- |
| MiniCPM-V | ✅ | ❌ | ❌ | 指南待补充 |
| MiMo-VL | ✅ | ❌ | ❌ | 指南待补充 |
| Qwen2.5-VL | ✅ | ✅ | ✅ | 指南待补充 |
| Qwen3-VL | ✅ | ✅ | ✅ | 指南待补充 |
| GLM-4.6V | ✅ | ❌ | ❌ | 指南待补充 |
| GLM-5.3-Flash-VL | ✅ | — | — | [预览版指南 ↗](/zh/cookbook/autoregressive_models/glm/glm_5_3_flash/) |
| VLM-R1 | ✅ | ❌ | ❌ | 指南待补充 |

</div>

</section>

<section data-model-category="rerank">

## Rerank

用于检索结果打分与排序的模型。

<div class="catalog-table" tabindex="0" role="region" aria-label="重排序">

| 模型 / 系列 | Ascend NPU | Cambricon MLU | Hygon DCU | 部署指南 |
| :--- | :---: | :---: | :---: | :--- |
| Qwen3-Reranker | ✅ | ❌ | ❌ | 指南待补充 |

</div>

</section>

<section data-model-category="dit">

## DiT

用于生成图像和视频的扩散模型。

<div class="catalog-table" tabindex="0" role="region" aria-label="图像与视频生成">

| 模型 / 系列 | Ascend NPU | Cambricon MLU | Hygon DCU | 部署指南 |
| :--- | :---: | :---: | :---: | :--- |
| Flux | ✅ | ❌ | ❌ | 指南待补充 |
| Flux2 | ✅ | — | — | [部署指南 ↗](/zh/cookbook/diffusion_models/flux/flux2/) |
| Wan2.2 | ✅ | — | — | [部署指南 ↗](/zh/cookbook/diffusion_models/wan/wan2_2/) |

</div>

</section>

<section data-model-category="rec">

## Rec

用于生成式推荐任务的模型。

<div class="catalog-table" tabindex="0" role="region" aria-label="生成式推荐">

| 模型 / 系列 | Ascend NPU | Cambricon MLU | Hygon DCU | 部署指南 |
| :--- | :---: | :---: | :---: | :--- |
| OneRec | ✅ | ❌ | ❌ | 指南待补充 |
| Qwen2 | ✅ | ❌ | ❌ | 指南待补充 |
| Qwen2.5 | ✅ | ❌ | ❌ | 指南待补充 |
| Qwen3 | ✅ | ❌ | ❌ | 指南待补充 |

</div>

</section>
