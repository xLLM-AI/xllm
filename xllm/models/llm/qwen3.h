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

#include <torch/nn/functional/normalization.h>

#include <type_traits>

#include "core/framework/model/model_output.h"
#include "core/util/rec_model_utils.h"
#if defined(USE_NPU)
#include "core/layers/common/attention_mask.h"
#endif
#include "core/layers/qwen3_decoder_layer.h"
#include "models/llm/llm_model_base.h"

namespace xllm {

class QWen3ModelImpl : public LlmModelImplBase<layer::Qwen3DecoderLayer> {
 public:
  QWen3ModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::Qwen3DecoderLayer>("qwen3",
                                                   context.get_model_args()) {
    // register submodules
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    if (!mrope_section_.empty()) {
      cos_sin_ = layer::rotary::get_concat_rotary_embedding(
          128,
          model_args.max_position_embeddings(),
          model_args.rope_theta(),
          options);
    }

    layers_.reserve(model_args.n_layers());
    norm_ = register_module("norm", layer::RMSNorm(context));
    // DFlash2 consumes the target model's embedding table.  Do not allocate a
    // draft-local copy here: set_word_embedding() binds the shared target
    // module after both workers finish loading.  Merely replacing the holder
    // after register_module() would leave the original large-vocabulary module
    // owned by torch::nn::Module and keep its device memory alive.
    if (!is_dflash2_draft_model_type(model_args.model_type())) {
      embed_tokens_ =
          register_module("embed_tokens", layer::WordEmbedding(context));
    }
#if defined(USE_NPU)
    attn_mask_ = layer::AttentionMask(
        options.device(), options.dtype().toScalarType(), /*mask_value=*/-9984);
    dense_attn_mask_ = layer::AttentionMask(
        options.device(), options.dtype().toScalarType(), /*mask_value=*/1);
#endif
    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      auto layer = layer::Qwen3DecoderLayer(context, i);
      layers_.push_back(layer);
    }
  }

  std::pair<torch::Tensor, torch::Tensor> apply_mrope(
      const torch::Tensor positions) override {
    auto target_cos_sin = cos_sin_.index({positions});
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();
    auto apply = [this](torch::Tensor x) {
      auto freqs_t = x[0].clone();
      // mrop_length == freqs_length == head_dim / 2
      int64_t mrop_length = static_cast<int64_t>(freqs_t.size(-1) / 2);

      for (int dim_idx = 1; dim_idx <= 2; ++dim_idx) {
        int64_t offset = dim_idx;
        int64_t section_len = mrope_section_[dim_idx];
        int64_t length = section_len * 3;

        // Since the last dim of freqs is repeated to 2*mrop_length
        // idx_first_half: [offset, offset+3, offset+6, ... < mrop_length]
        // idx_second_half: [mrop_length+offset, mrop_length+offset+3,
        //     mrop_length+offset+6, ... < 2*mrop_length]
        auto idx_first_half = torch::arange(offset, length, 3, torch::kLong);
        auto idx_second_half = torch::arange(
            offset + mrop_length, length + mrop_length, 3, torch::kLong);

        auto idx_tensor =
            torch::cat({idx_first_half, idx_second_half}, 0).to(x.device());
        // freqs_t[..., idx] = freqs[dim_idx][..., idx]
        auto src = x[dim_idx].index_select(-1, idx_tensor);
        freqs_t.index_copy_(-1, idx_tensor, src);
      }
      return freqs_t;
    };
    cos_pos = apply(cos_pos.reshape({positions.size(0), -1, cos_pos.size(-1)}));
    sin_pos = apply(sin_pos.reshape({positions.size(0), -1, sin_pos.size(-1)}));
    return std::make_pair(cos_pos, sin_pos);
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelParamsView& input_params) override {
    auto local_params = input_params.clone().view();
    return forward_impl(tokens, positions, kv_caches, local_params);
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const RecModelParams& input_params) {
    auto local_params = input_params.clone();
    return forward_impl(tokens, positions, kv_caches, local_params);
  }

  template <typename InputParams>
  ModelOutput forward_impl(torch::Tensor tokens,
                           torch::Tensor positions,
                           std::vector<KVCache>& kv_caches,
                           InputParams& input_params) {
    auto& input_params_new = input_params;

    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
    }
    auto inputs_embeds = input_params.embedding.input_embedding;
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = embed_tokens_(tokens);
    }

    auto& dp_token_nums = input_params_new.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);
    if (!input_params_new.attn_metadata) {
      input_params_new.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              get_attention_metadata(input_params_new, h));
    }

    auto& attn_metadata = *(input_params_new.attn_metadata);
    bool only_prefill =
        (attn_metadata.is_prefill || attn_metadata.is_chunked_prefill);
    if (positions.dim() == 2 && only_prefill && !mrope_section_.empty()) {
      std::tie(attn_metadata.mrope_cos, attn_metadata.mrope_sin) =
          apply_mrope(positions);
    }

    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < layers_.size(); i++) {
      if (!input_params_new.synchronize_layer(static_cast<uint32_t>(i))) {
        return ModelOutput();
      }
      prepare_qwen_rec_layer(
          input_params_new, attn_metadata, i, layers_.size());
#if defined(USE_CUDA)
      attn_metadata.plan_info->layer_id = i;
      if (attn_metadata.shared_plan_info != nullptr) {
        attn_metadata.shared_plan_info->layer_id = i;
      }
      if (attn_metadata.unshared_plan_info != nullptr) {
        attn_metadata.unshared_plan_info->layer_id = i;
      }
#endif
      auto& layer = layers_[i];
      h = layer->forward(h,
                         residual,
                         positions,
                         attn_metadata,
                         kv_caches[i],
                         input_params_new);
      if (!input_params_new.record_layer(static_cast<uint32_t>(i),
                                         h.device())) {
        return ModelOutput();
      }

      if constexpr (std::is_same_v<InputParams, ModelParamsView>) {
        if (input_params.has_multimodal()) {
          const auto& deep_stacks = input_params.multimodal().deep_stacks;
          if (i < deep_stacks.size()) {
            h = h + deep_stacks[i];
          }
        }
      }
    }
    auto [hidden_states, residual_out] = norm_(h, residual);
    return ModelOutput(hidden_states, residual_out);
  }

 protected:
  virtual layer::AttentionMetadata get_attention_metadata(
      const ModelParamsView& params,
      const torch::Tensor& h) {
    return build_attention_metadata(params, h);
  }

  layer::AttentionMetadata get_attention_metadata(const RecModelParams& params,
                                                  const torch::Tensor& h) {
    return build_attention_metadata(params, h);
  }

  template <typename InputParams>
  layer::AttentionMetadata build_attention_metadata(const InputParams& params,
                                                    const torch::Tensor& h) {
#if defined(USE_NPU)
    if (params.graph.attn_mask.defined()) {
      return layer::AttentionMetadataBuilder::build(
          params, model_args_.enable_mla(), params.graph.attn_mask);
    }
    max_seq_len_ = std::max(params.meta.kv_max_seq_len, max_seq_len_);
    bool is_spec_verify = false;
    if constexpr (std::is_same_v<InputParams, ModelParamsView>) {
      is_spec_verify = params.is_spec_verify;
    }
    const bool use_append_mask =
        is_spec_verify || params.meta.batch_forward_type.is_mixed() ||
        params.meta.batch_forward_type.is_chunked_prefill();
    if (!use_append_mask) {
      return layer::AttentionMetadataBuilder::build(
          params,
          model_args_.enable_mla(),
          dense_attn_mask_.get_attn_mask(
              max_seq_len_, h.dtype().toScalarType(), h.device()));
    }
    const int32_t num_sequences = params.meta.num_sequences;
    if (num_sequences <= 0) {
      return layer::AttentionMetadataBuilder::build(
          params,
          model_args_.enable_mla(),
          dense_attn_mask_.get_attn_mask(
              max_seq_len_, h.dtype().toScalarType(), h.device()));
    }

    std::vector<torch::Tensor> req_mask_vec;
    req_mask_vec.reserve(num_sequences);
    const int32_t max_kv_seq = params.meta.kv_max_seq_len;
    for (int32_t j = 0; j < num_sequences; ++j) {
      const int32_t q_len = params.attention.host.q_seq_lens[j];
      const int32_t kv_len = params.attention.host.kv_seq_lens[j];
      req_mask_vec.emplace_back(gen_append_attn_mask(
          q_len, kv_len, max_kv_seq, h.dtype().toScalarType(), h.device()));
    }
    return layer::AttentionMetadataBuilder::build(
        params, model_args_.enable_mla(), torch::cat(req_mask_vec, 0));
#else
    return layer::AttentionMetadataBuilder::build(params,
                                                  model_args_.enable_mla());
#endif
  }

#if defined(USE_NPU)
  virtual torch::Tensor gen_append_attn_mask(int32_t q_len,
                                             int32_t kv_len,
                                             int32_t max_kv_len,
                                             torch::Dtype dtype,
                                             torch::Device device) {
    return attn_mask_.gen_append_mask(q_len, kv_len, max_kv_len, dtype, device);
  }
#endif

 private:
#if defined(USE_NPU)
  layer::AttentionMask attn_mask_;
  layer::AttentionMask dense_attn_mask_;
#endif
};
TORCH_MODULE(QWen3Model);

class QWen3ForCausalLMImpl : public LlmForCausalLMImplBase<QWen3Model> {
 public:
  QWen3ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<QWen3Model>(context) {}

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
TORCH_MODULE(QWen3ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(qwen3, QWen3ForCausalLM);

REGISTER_CAUSAL_REC_MODEL(qwen3, QWen3ForCausalLM);

// register the model args
REGISTER_MODEL_ARGS(qwen3, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(dtype, "dtype", args->dtype());
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  // LOAD_ARG_OR(no_bias, "no_bias", true);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  [&] {
    const nlohmann::json data = json.data();
    std::vector<int32_t> eos_token_ids;
    if (data.contains("eos_token_id")) {
      const auto& eos_token_id = data.at("eos_token_id");
      if (eos_token_id.is_array()) {
        eos_token_ids = eos_token_id.get<std::vector<int32_t>>();
      } else if (eos_token_id.is_number_integer()) {
        eos_token_ids.push_back(eos_token_id.get<int32_t>());
      }
    }
    if (eos_token_ids.empty()) {
      eos_token_ids.push_back(151643);
    }
    args->eos_token_id_vec() = eos_token_ids;
    args->eos_token_id() = eos_token_ids.front();
  }();
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(rope_theta, "rope_parameters.rope_theta", args->rope_theta());
  LOAD_ARG_OR(rope_scaling_rope_type, "rope_scaling.rope_type", "");
  LOAD_ARG_OR(rope_scaling_factor, "rope_scaling.factor", 0.0f);
  LOAD_ARG_OR(rope_scaling_original_max_position_embeddings,
              "rope_scaling.original_max_position_embeddings",
              0);
  LOAD_ARG_OR(rope_scaling_beta_fast, "rope_scaling.beta_fast", 32);
  LOAD_ARG_OR(rope_scaling_beta_slow, "rope_scaling.beta_slow", 1);
  LOAD_ARG_OR(rope_scaling_attn_factor, "rope_scaling.attention_factor", 0.0f);

  // For qwen3/2.5 model < 7B,  tie_word_embeddings = true
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", -1);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);

  LOAD_ARG_OR(markov_rank, "markov_rank", 0);
  LOAD_ARG_OR(enable_confidence_head, "enable_confidence_head", false);
  LOAD_ARG_OR(
      confidence_head_with_markov, "confidence_head_with_markov", false);
  // Key fallback order: "dflash_config.*" container first, then the flat
  // top-level field, then the legacy "dflash2_*" spelling.
  LOAD_ARG_OR_FUNC(dflash2_block_size, "dflash_config.block_size", [&] {
    return json.value_or<int32_t>(
        std::vector<std::string>{"block_size", "dflash2_block_size"}, 0);
  });
  LOAD_ARG_OR_FUNC(
      dflash2_conv_group_size, "dflash_config.conv_group_size", [&] {
        return json.value_or<int32_t>(
            std::vector<std::string>{"conv_group_size",
                                     "dflash2_conv_group_size"},
            0);
      });
  LOAD_ARG_OR_FUNC(
      dflash2_conv_kernel_size, "dflash_config.conv_kernel_size", [&] {
        return json.value_or<int32_t>(
            std::vector<std::string>{"conv_kernel_size",
                                     "dflash2_conv_kernel_size"},
            0);
      });
  LOAD_ARG_OR_FUNC(dflash2_selector_rank, "dflash_config.selector_rank", [&] {
    return json.value_or<int32_t>(
        std::vector<std::string>{"selector_rank", "dflash2_selector_rank"}, 0);
  });
  LOAD_ARG_OR_FUNC(dflash2_selector_top_k, "dflash_config.selector_top_k", [&] {
    return json.value_or<int32_t>(
        std::vector<std::string>{"selector_top_k", "dflash2_selector_top_k"},
        0);
  });

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>(args->eos_token_id_vec().begin(),
                                      args->eos_token_id_vec().end()));
});

}  // namespace xllm
