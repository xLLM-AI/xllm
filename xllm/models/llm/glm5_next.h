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

#include "models/model_registry.h"
// Only the config-args loader (REGISTER_MODEL_ARGS) is needed for the
// --model_impl=python path; deliberately do NOT pull deepseek_v32.h (which,
// on NPU, transitively requires the NPU-only DeepseekV2DecoderLayer).

namespace xllm {

// glm5_next is served via --model_impl=python (PyCausalLM + the pure-torch
// xllm.python.models.glm5_next.Glm5NextForCausalLM, or Glm5NextVLModel
// for
// --backend=vlm). No C++ model class is needed; this header only registers the
// config.json -> ModelArgs loader so PyCausalLM::build_config_dict forwards
// the right fields to the python model.
// KDA (linear_attn_config: gate_lower_bound/safe_gate/full_attn_layers) and mHC
// (hc_mult/hc_eps/hc_sinkhorn_iters) fields are intentionally NOT plumbed here:
// the python Glm5NextConfig.from_dict defaults match the real 300B config, so
// the python side resolves them correctly without C++ plumbing.
// VLM note: the multimodal token ids and vision_config.* fields ARE plumbed
// (see the VLM block below) so the C++ multimodal processor
// (GLM4VPromptProcessor, registered in vlm/glm5_next_vlm.h) resolves GLM
// image token ids / merge size, and the Python GlmOcr ViT (glm5_next_vl.py)
// receives the real vision dims instead of its (different) defaults.

namespace glm5_next_args {

// Shared config.json -> ModelArgs loader for glm5_next and the
// glm5_next_mtp draft (same field set; the draft config.json only differs in
// num_hidden_layers / layer schedules / model_type).
inline bool load_model_args(const JsonReader& json, ModelArgs* args) {
  LOAD_ARG_OR(model_type, "model_type", "glm5_next");
  // The export tool writes "glm5_next" / "glm5_next_mtp" as model_type for
  // every GLM-5.3-Flash checkpoint (verified across GLM-5.3-Flash-W8A8,
  // GLM-next-w8a8, *-mtp). xllm registers under the same names, so no
  // normalization is needed; downstream model_type checks
  // (e.g. is_glm5_next_mtp_draft_model_type) match the checkpoint value
  // directly.
  // GLM-5.3-Flash config.json nests every text-side field under "text_config"
  // (verified against GLM-5.3-Flash-W8A8). Read those fields with an explicit
  // "text_config." prefix; do not rely on an implicit fallback in JsonReader.
  LOAD_ARG_OR(dtype, "text_config.torch_dtype", "bfloat16");
  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 154880);
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 4096);
  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 45);
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 64);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 64);
  LOAD_ARG_OR(intermediate_size, "text_config.intermediate_size", 12288);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 1104096);
  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-5);
  // fake config carries eos_token_id as a single-element array [154879]
  // (the C++ loader expects an array; the generator emits it that way).
  LOAD_ARG_OR_FUNC(eos_token_id_vec, "text_config.eos_token_id", [&] {
    return std::vector<int32_t>{154879};
  });
  LOAD_ARG_OR(bos_token_id, "text_config.bos_token_id", 0);
  LOAD_ARG_OR(rope_theta, "text_config.rope_parameters.rope_theta", 10000.0f);

  // MoE parameters
  LOAD_ARG_OR(first_k_dense_replace, "text_config.first_k_dense_replace", 3);
  LOAD_ARG_OR(hidden_act, "text_config.hidden_act", "silu");
  LOAD_ARG_OR(n_routed_experts, "text_config.n_routed_experts", 288);
  LOAD_ARG_OR(n_shared_experts, "text_config.n_shared_experts", 1);
  LOAD_ARG_OR(num_experts_per_tok, "text_config.num_experts_per_tok", 8);
  LOAD_ARG_OR(moe_intermediate_size, "text_config.moe_intermediate_size", 2048);
  LOAD_ARG_OR(routed_scaling_factor, "text_config.routed_scaling_factor", 2.5f);
  LOAD_ARG_OR(norm_topk_prob, "text_config.norm_topk_prob", true);
  LOAD_ARG_OR(n_group, "text_config.n_group", 1);
  LOAD_ARG_OR(topk_group, "text_config.topk_group", 1);

  // MLA (NoPE: qk_rope=0) + DSA indexer parameters
  LOAD_ARG_OR(qk_nope_head_dim, "text_config.qk_nope_head_dim", 256);
  LOAD_ARG_OR(qk_rope_head_dim, "text_config.qk_rope_head_dim", 0);
  LOAD_ARG_OR(v_head_dim, "text_config.v_head_dim", 256);
  LOAD_ARG_OR(q_lora_rank, "text_config.q_lora_rank", 1536);
  LOAD_ARG_OR(kv_lora_rank, "text_config.kv_lora_rank", 512);
  LOAD_ARG_OR(index_head_dim, "text_config.index_head_dim", 128);
  LOAD_ARG_OR(index_n_heads, "text_config.index_n_heads", 32);
  LOAD_ARG_OR(index_topk, "text_config.index_topk", 2048);
  LOAD_ARG_OR(index_kpool, "text_config.index_kpool", 1);
  LOAD_ARG_OR(index_kpool_compress, "text_config.index_kpool_compress", false);
  LOAD_ARG_OR(index_kpool_always_select_tail,
              "text_config.index_kpool_always_select_tail",
              false);
  LOAD_ARG_OR(index_topk_freq, "text_config.index_topk_freq", 1);
  LOAD_ARG_OR(index_skip_topk_offset, "text_config.index_skip_topk_offset", 1);
  LOAD_ARG_OR(
      indexer_types, "text_config.indexer_types", std::vector<std::string>());
  LOAD_ARG_OR(mlp_layer_types,
              "text_config.mlp_layer_types",
              std::vector<std::string>());

  // KDA linear-attention config (transformers nests it under
  // linear_attn_config). Drives the engine's linear-state slot pool:
  // has_linear_attention_layers(args) becomes true, and conv/ssm caches
  // are allocated for KDA layers (see is_linear_attention_layer).
  LOAD_ARG_OR(full_attn_layers,
              "text_config.linear_attn_config.full_attn_layers",
              std::vector<int32_t>{});
  LOAD_ARG_OR(linear_num_key_heads,
              "text_config.linear_attn_config.num_heads",
              args->linear_num_key_heads());
  LOAD_ARG_OR(linear_key_head_dim,
              "text_config.linear_attn_config.head_dim",
              args->linear_key_head_dim());
  LOAD_ARG_OR_FUNC(linear_num_value_heads,
                   "text_config.linear_attn_config.num_heads",
                   [&] { return args->linear_num_key_heads(); });
  LOAD_ARG_OR_FUNC(linear_value_head_dim,
                   "text_config.linear_attn_config.head_dim",
                   [&] { return args->linear_key_head_dim(); });
  LOAD_ARG_OR(linear_conv_kernel_dim,
              "text_config.linear_attn_config.short_conv_kernel_size",
              args->linear_conv_kernel_dim());
  // recurrent_state must be fp32 for numerical stability.
  LOAD_ARG_OR(mamba_ssm_dtype, "text_config.mamba_ssm_dtype", "float32");
  // layer_types mirrors the python _resolve_schedules derivation so C++ and
  // python agree on which layers are KDA (linear) vs DSA (full attention).
  LOAD_ARG_OR(layer_types, "text_config.layer_types", args->layer_types());
  // When config.json omits layer_types, derive from full_attn_layers:
  // a layer is full-attention iff its index is in full_attn_layers. When
  // full_attn_layers is also absent, fall back to every 4th layer (i % 4 == 3)
  // mirroring the Python from_dict fallback so C++ and Python stay consistent
  // even for a config that omits both fields (all real GLM5.3 checkpoints set
  // both, so this branch is not reached in practice).
  if (args->layer_types().empty()) {
    std::vector<int32_t> fa = args->full_attn_layers();
    if (fa.empty()) {
      fa.reserve(static_cast<size_t>(args->n_layers()));
      for (int32_t i = 0; i < args->n_layers(); ++i) {
        if (i % 4 == 3) {
          fa.push_back(i);
        }
      }
      SET_ARG(full_attn_layers, fa);
    }
    std::vector<std::string> derived(static_cast<size_t>(args->n_layers()),
                                     "linear_attention");
    for (int32_t idx : fa) {
      if (idx >= 0 && idx < args->n_layers()) {
        derived[static_cast<size_t>(idx)] = "deepseek_sparse_attention";
      }
    }
    SET_ARG(layer_types, derived);
  }

  // VLM (vision) config + multimodal token ids. Plumbed so that
  // --backend=vlm --model_impl=python forwards the real GlmOcr vision
  // dims to the Python ViT (xllm.python.models.glm5_next_vl) via the flat
  // mm_-prefixed ModelArgs dict, and so the C++ multimodal processor
  // (GLM4VPromptProcessor, registered in vlm/glm5_next_vlm.h) resolves the
  // GLM-family image token ids and merge size. Defaults match the real
  // GLM-5.3-Flash-VL config so the ViT computes correct dims even if a field
  // is absent from config.json (the Python Glm5NextVisionConfig defaults
  // differ — e.g. image_size=336, out_hidden_size=1536 — and would
  // miscompute if relied upon).
  LOAD_ARG_OR(image_token_id, "image_token_id", 154854);
  LOAD_ARG_OR(video_token_id, "video_token_id", 154855);
  LOAD_ARG_OR(image_start_token_id, "image_start_token_id", 154830);
  LOAD_ARG_OR(image_end_token_id, "image_end_token_id", 154831);
  LOAD_ARG_OR(video_start_token_id, "video_start_token_id", 154832);
  LOAD_ARG_OR(video_end_token_id, "video_end_token_id", 154833);

  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 24);
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "silu");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1024);
  LOAD_ARG_OR(mm_image_size, "vision_config.image_size", 448);
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_channels", 3);
  LOAD_ARG_OR(mm_initializer_range, "vision_config.initializer_range", 0.02);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 4096);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 16);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.out_hidden_size", 4096);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  LOAD_ARG_OR(mm_layer_norm_eps, "vision_config.rms_norm_eps", 1e-5);
  LOAD_ARG_OR(mm_dropout, "vision_config.attention_dropout", 0.0f);
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);
  // GLM4VPromptProcessor / Qwen2VLImageProcessor read mm_image_merge_size
  // for the per-image token count (grid_thw.prod() / merge_size**2). The
  // HF loader also sets it from preprocessor_config.json; this explicit
  // load is a fallback so it is never 0 (which would divide by zero).
  LOAD_ARG_OR(mm_image_merge_size, "vision_config.spatial_merge_size", 2);

  // Computed / convention parameters
  SET_ARG(head_dim, args->qk_nope_head_dim() + args->qk_rope_head_dim());
  SET_ARG(rope_scaling_rope_type, "default");
  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>(args->eos_token_id_vec().begin(),
                                      args->eos_token_id_vec().end()));
  return true;
}

}  // namespace glm5_next_args

REGISTER_MODEL_ARGS_LOADER(glm5_next, &glm5_next_args::load_model_args);
REGISTER_MODEL_ARGS_LOADER(glm5_next_mtp, &glm5_next_args::load_model_args);

}  // namespace xllm
