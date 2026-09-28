/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#pragma once

// Multimodal-processor registration for glm5_next served as a VLM via
// --backend=vlm --model_impl=python (PyCausalLM + the pure-torch
// xllm.python.models.glm5_next_vl.Glm5NextVLModel).
//
// glm5_next has no C++ VLM model object (the language model and the GlmOcr
// vision tower both live in Python). The framework still requires a
// multimodal processor to satisfy VLMMaster's create_multimodal_processor
// CHECK, so this header registers one that:
//   1. Expands the text prompt with GLM-family image placeholders via
//      GLM4VPromptProcessor (token conventions: image_start_token_id /
//      image_end_token_id / image_token_id — identical to GLM-next's config,
//      and token count per image = grid_thw.prod() / merge_size**2, matching
//      the GlmOcr ViT's spatial-merge output count).
//   2. Delegates the actual image preprocessing to Python when
//      model_impl=python via the PyImagePreprocess branch wired into
//      Qwen2VLImageProcessor::process (HF AutoImageProcessor, the same
//      algorithm the Python ViT consumes).
//
// Reusing GLM4VPromptProcessor + Qwen2VLImageProcessor (rather than
// Glm4vMoeMultimodalProcessor) is deliberate: the latter binds
// Glm4VImageProcessor, which lacks the model_impl=python delegation branch.

#include "models/model_registry.h"
#include "processors/glm4v_prompt_processor.h"
#include "processors/glm4v_video_processor.h"
#include "processors/multimodal_processor.h"
#include "processors/qwen2_vl_image_processor.h"

namespace xllm {

// Image preprocessing delegates to Python (Qwen2VLImageProcessor -> HF
// AutoImageProcessor); video preprocessing uses the GLM-family
// Glm4VVideoProcessor (frame sampling / resize / normalize / patchify ->
// pixel_values_videos + video_grid_thw), which GLM4VPromptProcessor then
// expands into video_start/video_token/video_end placeholders. The two
// masks are disjoint, so image and video coexist in one batch.
using Glm5NextVLMultimodalProcessor = MultimodalProcessor<GLM4VPromptProcessor,
                                                          Qwen2VLImageProcessor,
                                                          Glm4VVideoProcessor>;

REGISTER_MULTIMODAL_PROCESSOR(glm5_next, Glm5NextVLMultimodalProcessor);

}  // namespace xllm
