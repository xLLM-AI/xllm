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

class Qwen3_VLMoeForConditionalGenerationImpl
    : public Qwen3VLForConditionalGenerationBase<Qwen3_VisionTransformer,
                                                 Qwen3MoeForCausalLM> {
 public:
  Qwen3_VLMoeForConditionalGenerationImpl(const ModelContext& context)
      : Qwen3VLForConditionalGenerationBase(context) {}
};
TORCH_MODULE(Qwen3_VLMoeForConditionalGeneration);

using Qwen3VLMoeMultimodalProcessor =
    MultimodalProcessor<Qwen3VLPromptProcessor,
                        Qwen2VLImageProcessor,
                        Qwen3VLVideoProcessor>;
REGISTER_MULTIMODAL_PROCESSOR(qwen3_vl_moe, Qwen3VLMoeMultimodalProcessor);
REGISTER_CAUSAL_VLM_MODEL(qwen3_vl_moe, Qwen3_VLMoeForConditionalGeneration);
REGISTER_MPOSITION_GENERATOR(qwen3_vl_moe, Qwen3VLMPositionGenerator);

// register the model args
REGISTER_MODEL_ARGS(qwen3_vl_moe, [&] {
  // text config
  LOAD_ARG_OR(model_type, "model_type", "qwen3_vl_moe");
  LOAD_ARG_OR(attention_bias, "text_config.attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(bos_token_id, "text_config.bos_token_id", 151643);
  LOAD_ARG_OR(decoder_sparse_step, "text_config.decoder_sparse_step", 1);
  LOAD_ARG_OR(dtype, "text_config.dtype", "bfloat16");
  LOAD_ARG_OR(eos_token_id, "text_config.eos_token_id", 151645);
  LOAD_ARG_OR_FUNC(head_dim, "text_config.head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
  LOAD_ARG_OR(hidden_act, "text_config.hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 2048);
  LOAD_ARG_OR(initializer_range, "text_config.initializer_range", 0.02);
  LOAD_ARG_OR(intermediate_size, "text_config.intermediate_size", 5632);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 128000);
  // LOAD_ARG(mlp_only_layers, "text_config.mlp_only_layers");
  LOAD_ARG_OR(moe_intermediate_size, "text_config.moe_intermediate_size", 1408);
  LOAD_ARG_OR(norm_topk_prob, "text_config.norm_topk_prob", true);
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 16);
  LOAD_ARG_OR(num_experts, "text_config.num_experts", 128);
  LOAD_ARG_OR(num_experts_per_tok, "text_config.num_experts_per_tok", 8);
  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 24);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 16);
  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-06);
  LOAD_ARG_OR(rope_scaling_rope_type, "text_config.rope_scaling.type", "mrope");
  LOAD_ARG(rope_scaling_mrope_section,
           "text_config.rope_scaling.mrope_section");
  // LOAD_ARG_OR(rope_scaling_mrope_interleaved,"text_config.rope_scaling.mrope_interleaved",true);
  LOAD_ARG_OR(rope_theta, "text_config.rope_theta", 5000000.0f);
  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 151936);

  // vision config
  LOAD_ARG(mm_deepstack_visual_indexes,
           "vision_config.deepstack_visual_indexes");
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 27);
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "gelu_pytorch_tanh");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1152);
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_channels", 3);
  LOAD_ARG_OR(mm_initializer_range, "vision_config.initializer_range", 0.02);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 4304);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 16);
  LOAD_ARG_OR(mm_num_position_embeddings,
              "vision_config.num_position_embeddings",
              2304);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.out_hidden_size", 3584);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 16);
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);
  // Read the vision-scoped key so the top-level fallback in JsonReader does
  // not resolve "head_dim" against text_config: Qwen3-VL keeps the text
  // head_dim (128) under text_config and derives the vision head_dim from
  // vision_config.hidden_size / vision_config.num_heads (1152 / 16 = 72).
  LOAD_ARG_OR_FUNC(mm_head_dim, "vision_config.head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });

  LOAD_ARG_OR(image_token_id, "image_token_id", 151655);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(video_token_id, "video_token_id", 151656);
  LOAD_ARG_OR(vision_end_token_id, "vision_end_token_id", 151653);
  LOAD_ARG_OR(vision_start_token_id, "vision_start_token_id", 151652);

  // arguments to be compatible with other fused moe models
  LOAD_ARG_OR(n_routed_experts, "num_experts", 128);
  SET_ARG(n_shared_experts, 0);
  SET_ARG(scoring_func, "softmax");
  SET_ARG(topk_method, "");
  SET_ARG(n_group, -1);
  SET_ARG(topk_group, 0);
  SET_ARG(routed_scaling_factor, 1.0);
});
}  // namespace xllm
