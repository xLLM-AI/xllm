/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "core/framework/model/model_output.h"
#include "core/layers/common/lm_head.h"
#include "models/model_registry.h"
#include "models/vlm/mposition/mposition.h"
#include "models/vlm/qwen3_vl_base.h"
#include "processors/multimodal_processor.h"
#include "processors/qwen2_vl_image_processor.h"
#include "processors/qwen3_vl_prompt_processor.h"
#include "processors/qwen3_vl_video_processor.h"

#if defined(USE_NPU)
#include "models/llm/qwen3_5.h"
#include "models/vlm/npu/qwen3_vl.h"
#elif defined(USE_MLU) || defined(USE_DCU)
#include "core/layers/common/qwen3_next_rms_norm.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/qwen3_5_decoder_layer.h"
#include "core/layers/qwen3_vision_layer.h"
#include "models/llm/llm_model_base.h"
#include "qwen3_vl.h"
#endif

namespace xllm {
#if !defined(USE_NPU)

class Qwen3_5ModelImpl final
    : public LlmModelImplBase<layer::Qwen3_5DecoderLayer> {
 public:
  Qwen3_5ModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::Qwen3_5DecoderLayer>("qwen3_5",
                                                     context.get_model_args()) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();
    dp_size_ = parallel_args.dp_size();

    if (!mrope_section_.empty()) {
      int64_t rotary_dim = static_cast<int64_t>(
          model_args.head_dim() * model_args.partial_rotary_factor());
      cos_sin_ = layer::rotary::get_concat_rotary_embedding(
          rotary_dim,
          model_args.max_position_embeddings(),
          model_args.rope_theta(),
          options);
    }

    layers_.reserve(model_args.n_layers());
    rms_norm_ = register_module(
        "norm",
        layer::Qwen3NextRMSNorm(
            model_args.hidden_size(), model_args.rms_norm_eps(), options));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));

    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      auto layer = layer::Qwen3_5DecoderLayer(context, i);
      layers_.push_back(layer);
    }
  }

  void load_state_dict(const StateDict& state_dict) override {
    embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));

    // call each layer's load_state_dict function
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    rms_norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  std::pair<torch::Tensor, torch::Tensor> apply_mrope(
      const torch::Tensor positions) override {
    auto target_cos_sin = cos_sin_.index({positions});
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();
    auto options = positions.options().dtype(torch::kLong);
    auto apply = [this, options](torch::Tensor x) {
      auto freqs_t = x[0].clone();
      int64_t mrop_length = static_cast<int64_t>(freqs_t.size(-1) / 2);

      for (int32_t dim_idx = 1; dim_idx <= 2; ++dim_idx) {
        int64_t offset = dim_idx;
        int64_t section_len = mrope_section_[dim_idx];
        int64_t length = section_len * 3;

        auto idx_first_half = torch::arange(offset, length, 3, options);
        auto idx_second_half = torch::arange(
            offset + mrop_length, length + mrop_length, 3, options);

        auto idx_tensor =
            torch::cat({idx_first_half, idx_second_half}, 0).to(x.device());
        auto src = x[dim_idx].index_select(-1, idx_tensor);
        freqs_t.index_copy_(-1, idx_tensor, src);
      }
      return freqs_t;
    };
    cos_pos = apply(cos_pos.reshape({positions.size(0), -1, cos_pos.size(-1)}));
    sin_pos = apply(sin_pos.reshape({positions.size(0), -1, sin_pos.size(-1)}));
    return std::make_pair(cos_pos, sin_pos);
  }

  virtual ModelOutput forward(torch::Tensor tokens,
                              torch::Tensor positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) {
    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    std::vector<torch::Tensor> deep_stacks;

    if (dp_size_ > 1) {
      if (tokens.numel() == 0) {
        tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
        positions = torch::tensor({1}).to(torch::kInt32).to(positions.device());
      }
      auto& dp_token_nums = input_params_new.parallel.dp_global_token_nums;
      std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);
    }

    auto inputs_embeds = input_params.embedding.input_embedding;
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = embed_tokens_(tokens);
    }

    if (!input_params_new.attn_metadata) {
      input_params_new.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              get_attention_metadata(input_params_new, h));
    }

    auto& attn_metadata = *(input_params_new.attn_metadata);
    std::tie(attn_metadata.mrope_cos, attn_metadata.mrope_sin) =
        apply_mrope(positions);

    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h,
                residual,
                positions,
                attn_metadata,
                kv_caches[i],
                input_params_new);
    }
    if (residual.has_value()) {
      h = h + residual.value();
    }
    auto hidden_states = std::get<0>(rms_norm_(h));
    return ModelOutput(hidden_states);
  }

 private:
  int32_t dp_size_ = 1;
  layer::Qwen3NextRMSNorm rms_norm_{nullptr};
  layer::AttentionMetadata get_attention_metadata(
      const ModelInputParams& params,
      const torch::Tensor& h) {
    auto attn_metadata =
        layer::AttentionMetadataBuilder::build(params, /*enable_mla=*/false);
    // Init batch and token_block_offset for GDN attention
    if (attn_metadata.is_prefill || attn_metadata.is_chunked_prefill) {
      constexpr int32_t kBlockM = 64;
      constexpr int64_t pad_slot_id = -1;
      constexpr int64_t default_max_num_programs = 1024;
      constexpr int64_t chunk_size = 64;
      auto seqlens = attn_metadata.q_cu_seq_lens.diff();
      auto nums = (seqlens + kBlockM - 1) / kBlockM;
      nums = nums.to(torch::kLong);
      int32_t tot = nums.sum().item<int32_t>();
      torch::Tensor range_batch = torch::arange(nums.size(0), nums.options());
      torch::Tensor mlist_tensor = torch::repeat_interleave(range_batch, nums);
      int64_t mlist_len = mlist_tensor.size(0);
      int64_t max_num_programs =
          std::max(default_max_num_programs, mlist_len) * 2;
      torch::Tensor batch_ptr =
          torch::full({max_num_programs},
                      pad_slot_id,
                      torch::dtype(torch::kInt32).device(seqlens.device()));
      torch::Tensor token_block_offset_ptr =
          torch::full({max_num_programs},
                      pad_slot_id,
                      torch::dtype(torch::kInt32).device(seqlens.device()));

      std::vector<torch::Tensor> vec;
      vec.reserve(nums.size(0));
      for (int64_t i = 0; i < nums.size(0); ++i) {
        vec.emplace_back(
            torch::arange(nums[i].item<int64_t>(), nums.options()));
      }
      torch::Tensor offsetlist_tensor = torch::cat(vec, -1).to(torch::kInt32);
      batch_ptr.narrow(0, 0, mlist_len).copy_(mlist_tensor);
      token_block_offset_ptr.narrow(0, 0, mlist_len).copy_(offsetlist_tensor);

      // Compute chunk indices for the chunked GDN kernel
      {
        torch::Tensor lengths = seqlens;
        torch::Tensor num_chunks = (lengths + chunk_size - 1) / chunk_size;
        num_chunks = num_chunks.to(torch::kLong);
        torch::Tensor cumsum = torch::cumsum(num_chunks, 0);
        int64_t total_chunks = cumsum[-1].item<int64_t>();
        torch::Tensor arange_total =
            torch::arange(total_chunks, attn_metadata.q_cu_seq_lens.options());
        torch::Tensor zeros = torch::zeros({1}, cumsum.options());
        torch::Tensor prefix = torch::cat(
            {zeros, cumsum.slice(/*dim=*/0, /*start=*/0, /*end=*/-1)});
        torch::Tensor repeats_prefix =
            torch::repeat_interleave(prefix, num_chunks);
        torch::Tensor indices = arange_total - repeats_prefix;
        torch::Tensor mask = indices == 0;
        torch::Tensor col0 = mask.cumsum(0) - 1;
        attn_metadata.chunk_indices = torch::stack({col0, indices}, /*dim=*/1)
                                          .to(attn_metadata.q_cu_seq_lens)
                                          .to(torch::kInt32);
      }
      attn_metadata.tot = tot;
      attn_metadata.batch = batch_ptr;
      attn_metadata.token_block_offset = token_block_offset_ptr;
      if (params.attention.device.kv_cache_tokens_nums.defined() &&
          params.attention.device.kv_cache_tokens_nums.numel() > 0) {
        attn_metadata.has_initial_states =
            (params.attention.device.kv_cache_tokens_nums > 0).to(torch::kBool);
      } else {
        attn_metadata.has_initial_states =
            torch::zeros({seqlens.size(0)},
                         torch::dtype(torch::kBool).device(seqlens.device()));
      }
    }
    return attn_metadata;
  }
};
TORCH_MODULE(Qwen3_5Model);

class Qwen3_5ForCausalLMImpl : public LlmForCausalLMImplBase<Qwen3_5Model> {
 public:
  Qwen3_5ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<Qwen3_5Model>(context) {}

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    namespace F = torch::nn::functional;
    return F::normalize(h, F::NormalizeFuncOptions().p(2).dim(1));
  }
};
TORCH_MODULE(Qwen3_5ForCausalLM);

#endif  // !defined(USE_NPU)

#if defined(USE_NPU)
using Qwen3_5_VisionTransformer = npu::model::Qwen3_VisionTransformer;
#else
using Qwen3_5_VisionTransformer = Qwen3_VisionTransformer;
#endif

using Qwen3_5ForConditionalGenerationImpl =
    Qwen3VLForConditionalGenerationBase<Qwen3_5_VisionTransformer,
                                        Qwen3_5ForCausalLM>;
TORCH_MODULE(Qwen3_5ForConditionalGeneration);

#define LOAD_QWEN3_5_COMMON_ARGS()                                             \
  LOAD_ARG_OR(model_type, "model_type", "qwen3_5");                            \
  LOAD_ARG_OR(dtype, "text_config.dtype", "bfloat16");                         \
  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 248320);                   \
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 5120);                   \
  LOAD_ARG_OR(hidden_act, "text_config.hidden_act", "silu");                   \
  LOAD_ARG_OR(intermediate_size, "text_config.intermediate_size", 17408);      \
  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 64);                  \
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 24);                 \
  LOAD_ARG(n_kv_heads, "text_config.num_key_value_heads");                     \
  LOAD_ARG_OR(                                                                 \
      max_position_embeddings, "text_config.max_position_embeddings", 262144); \
  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-6);                 \
  LOAD_ARG_OR(bos_token_id, "text_config.bos_token_id", 151643);               \
  LOAD_ARG_OR(eos_token_id, "text_config.eos_token_id", 248044);               \
  LOAD_ARG_OR(                                                                 \
      rope_theta, "text_config.rope_parameters.rope_theta", 10000000.0f);      \
  LOAD_ARG_OR(head_dim, "text_config.head_dim", 256);                          \
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);              \
  LOAD_ARG(layer_types, "text_config.layer_types");                            \
  LOAD_ARG_OR(                                                                 \
      linear_conv_kernel_dim, "text_config.linear_conv_kernel_dim", 4);        \
  LOAD_ARG_OR(linear_key_head_dim, "text_config.linear_key_head_dim", 128);    \
  LOAD_ARG_OR(                                                                 \
      linear_value_head_dim, "text_config.linear_value_head_dim", 128);        \
  LOAD_ARG_OR(linear_num_key_heads, "text_config.linear_num_key_heads", 16);   \
  LOAD_ARG_OR(linear_num_value_heads,                                          \
              "text_config.linear_num_value_heads",                            \
              static_cast<int32_t>(args->n_heads() * 2));                      \
  LOAD_ARG_OR(                                                                 \
      full_attention_interval, "text_config.full_attention_interval", 4);      \
  LOAD_ARG_OR(attn_output_gate, "text_config.attn_output_gate", true);         \
  LOAD_ARG_OR(                                                                 \
      num_nextn_predict_layers, "text_config.mtp_num_hidden_layers", 0);       \
  LOAD_ARG_OR(num_nextn_predict_layers,                                        \
              "text_config.num_nextn_predict_layers",                          \
              args->num_nextn_predict_layers());                               \
  LOAD_ARG_OR(attention_bias, "text_config.attention_bias", false);            \
  LOAD_ARG_OR(attention_dropout, "text_config.attention_dropout", 0.0f);       \
  LOAD_ARG_OR(initializer_range, "text_config.initializer_range", 0.02f);      \
  LOAD_ARG_OR(                                                                 \
      mlp_only_layers, "text_config.mlp_only_layers", std::vector<int32_t>()); \
  LOAD_ARG_OR(rope_scaling_mrope_section,                                      \
              "text_config.rope_parameters.mrope_section",                     \
              std::vector<int64_t>({11, 11, 10}));                             \
  LOAD_ARG_OR(rope_scaling_mrope_interleaved,                                  \
              "text_config.rope_parameters.mrope_interleaved",                 \
              true);                                                           \
  LOAD_ARG_OR(rope_scaling_rope_type,                                          \
              "text_config.rope_parameters.rope_type",                         \
              "default");                                                      \
  if (args->rope_scaling_rope_type() == "default") {                           \
    args->rope_scaling_rope_type() = "mrope";                                  \
  }                                                                            \
  LOAD_ARG_OR(partial_rotary_factor,                                           \
              "text_config.rope_parameters.partial_rotary_factor",             \
              0.25f);                                                          \
  LOAD_ARG_OR(mamba_ssm_dtype, "text_config.mamba_ssm_dtype", "float32")

#define LOAD_QWEN3_5_VISION_ARGS()                                             \
  LOAD_ARG_OR(image_token_id, "image_token_id", 248056);                       \
  LOAD_ARG_OR(video_token_id, "video_token_id", 248057);                       \
  LOAD_ARG_OR(vision_start_token_id, "vision_start_token_id", 248053);         \
  LOAD_ARG_OR(vision_end_token_id, "vision_end_token_id", 248054);             \
  LOAD_ARG_OR(mm_deepstack_visual_indexes,                                     \
              "vision_config.deepstack_visual_indexes",                        \
              std::vector<int64_t>());                                         \
  if (!args->mm_deepstack_visual_indexes().empty()) {                          \
    LOG(FATAL) << "qwen3_5 VLM does not support DeepStack visual indexes";     \
  }                                                                            \
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 27);                \
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "gelu_pytorch_tanh"); \
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1152);              \
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_channels", 3);                \
  LOAD_ARG_OR(mm_initializer_range, "vision_config.initializer_range", 0.02f); \
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 4304);  \
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 16);          \
  LOAD_ARG_OR(mm_num_position_embeddings,                                      \
              "vision_config.num_position_embeddings",                         \
              2304);                                                           \
  LOAD_ARG_OR(mm_projection_dim,                                               \
              "vision_config.out_hidden_size",                                 \
              args->hidden_size());                                            \
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 16);                  \
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);   \
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2); \
  LOAD_ARG_OR_FUNC(mm_head_dim, "head_dim", [&] {                              \
    return args->mm_hidden_size() / args->mm_num_attention_heads();            \
  })

// qwen3_5/qwen3_5_moe are multimodal entry points. On NPU, text-only serving
// uses qwen3_5_text/qwen3_5_moe_text from llm/qwen3_5.h because the VLM
// request protocol currently requires array-form chat content.
REGISTER_CAUSAL_VLM_MODEL(qwen3_5, Qwen3_5ForConditionalGeneration);
REGISTER_MPOSITION_GENERATOR(qwen3_5, Qwen3VLMPositionGenerator);
using Qwen35MultimodalProcessor = MultimodalProcessor<Qwen3VLPromptProcessor,
                                                      Qwen2VLImageProcessor,
                                                      Qwen3VLVideoProcessor>;
REGISTER_MULTIMODAL_PROCESSOR(qwen3_5, Qwen35MultimodalProcessor);
REGISTER_MODEL_ARGS(qwen3_5, [&] {
  LOAD_QWEN3_5_COMMON_ARGS();
  LOAD_QWEN3_5_VISION_ARGS();

  SET_ARG(num_experts, 0);
  SET_ARG(n_routed_experts, 0);
  SET_ARG(n_shared_experts, 0);

  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>({args->eos_token_id(), 248046}));
});

REGISTER_CAUSAL_VLM_MODEL(qwen3_5_moe, Qwen3_5ForConditionalGeneration);
REGISTER_MPOSITION_GENERATOR(qwen3_5_moe, Qwen3VLMPositionGenerator);
REGISTER_MULTIMODAL_PROCESSOR(qwen3_5_moe, Qwen35MultimodalProcessor);
REGISTER_MODEL_ARGS(qwen3_5_moe, [&] {
  LOAD_QWEN3_5_COMMON_ARGS();
  LOAD_QWEN3_5_VISION_ARGS();
  LOAD_ARG_OR(decoder_sparse_step, "text_config.decoder_sparse_step", 1);
  LOAD_ARG_OR(moe_intermediate_size, "text_config.moe_intermediate_size", 512);
  LOAD_ARG_OR(num_experts, "text_config.num_experts", 512);
  LOAD_ARG_OR(num_experts_per_tok, "text_config.num_experts_per_tok", 10);
  LOAD_ARG_OR(shared_expert_intermediate_size,
              "text_config.shared_expert_intermediate_size",
              512);
  LOAD_ARG_OR(norm_topk_prob, "text_config.norm_topk_prob", true);
  LOAD_ARG_OR(
      n_routed_experts, "text_config.n_routed_experts", args->num_experts());
  SET_ARG(n_shared_experts,
          args->shared_expert_intermediate_size() > 0 ? 1 : 0);
  SET_ARG(scoring_func, "softmax");
  SET_ARG(topk_method, "");
  SET_ARG(n_group, -1);
  SET_ARG(topk_group, 0);
  SET_ARG(routed_scaling_factor, 1.0f);

  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>({args->eos_token_id(), 248046}));
});

// Text-only model registrations. On NPU these are handled by llm/qwen3_5.h.
#if !defined(USE_NPU)
// qwen3_5 without vision config (text-only serving).
// Model args are already registered by the VLM registration above.
REGISTER_CAUSAL_MODEL_WITH_VARNAME(qwen3_5_lm, qwen3_5, Qwen3_5ForCausalLM);

REGISTER_CAUSAL_MODEL(qwen3_5_text, Qwen3_5ForCausalLM);
REGISTER_MODEL_ARGS(qwen3_5_text, [&] {
  LOAD_QWEN3_5_COMMON_ARGS();
  SET_ARG(num_experts, 0);
  SET_ARG(n_routed_experts, 0);
  SET_ARG(n_shared_experts, 0);
  SET_ARG(decoder_sparse_step, 1);
  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>({args->eos_token_id(), 248046}));
});

REGISTER_CAUSAL_MODEL(qwen3_5_moe_text, Qwen3_5ForCausalLM);
REGISTER_MODEL_ARGS(qwen3_5_moe_text, [&] {
  LOAD_QWEN3_5_COMMON_ARGS();
  LOAD_ARG_OR(decoder_sparse_step, "text_config.decoder_sparse_step", 1);
  LOAD_ARG_OR(moe_intermediate_size, "text_config.moe_intermediate_size", 512);
  LOAD_ARG_OR(num_experts, "text_config.num_experts", 512);
  LOAD_ARG_OR(num_experts_per_tok, "text_config.num_experts_per_tok", 10);
  LOAD_ARG_OR(shared_expert_intermediate_size,
              "text_config.shared_expert_intermediate_size",
              512);
  LOAD_ARG_OR(norm_topk_prob, "text_config.norm_topk_prob", true);
  LOAD_ARG_OR(
      n_routed_experts, "text_config.n_routed_experts", args->num_experts());
  SET_ARG(n_shared_experts,
          args->shared_expert_intermediate_size() > 0 ? 1 : 0);
  SET_ARG(scoring_func, "softmax");
  SET_ARG(topk_method, "");
  SET_ARG(n_group, -1);
  SET_ARG(topk_group, 0);
  SET_ARG(routed_scaling_factor, 1.0f);
  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>({args->eos_token_id(), 248046}));
});
#endif  // !defined(USE_NPU)

#undef LOAD_QWEN3_5_VISION_ARGS
#undef LOAD_QWEN3_5_COMMON_ARGS

}  // namespace xllm
