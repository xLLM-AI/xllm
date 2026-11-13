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

#include "core/framework/multimodal/mposition.h"
#include "models/llm/qwen3_moe.h"
#include "models/model_registry.h"
#include "models/vlm/qwen3_vl.h"
#include "models/vlm/qwen3_vl_base.h"
#include "processors/multimodal_processor_pipeline.h"
#include "processors/qwen2_vl_image_processor.h"
#include "processors/qwen3_vl_prompt_processor.h"
#include "processors/qwen3_vl_video_processor.h"

namespace xllm {

using Qwen3_VLMoeForConditionalGenerationImpl =
    Qwen3VLForConditionalGenerationBase<Qwen3_VisionTransformer,
                                        Qwen3MoeForCausalLM>;
TORCH_MODULE(Qwen3_VLMoeForConditionalGeneration);

using Qwen3VLMoeMultimodalProcessor =
    MultimodalProcessor<Qwen3VLPromptProcessor,
                        Qwen2VLImageProcessor,
                        Qwen3VLVideoProcessor>;
REGISTER_CAUSAL_MODEL(
    qwen3_vl_moe,
    Qwen3_VLMoeForConditionalGeneration,
    /*priority=*/0,
    ModelRegistry::make_multimodal_processor<Qwen3VLMoeMultimodalProcessor>);
REGISTER_MPOSITION_GENERATOR(qwen3_vl_moe, Qwen3VLMPositionGenerator);
REGISTER_MODEL_ARGS_LOADER(qwen3_vl_moe, kQwen3VLMoeArgsLoader);

}  // namespace xllm
