/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <torch_npu/csrc/core/npu/NPUFormat.h>

#include "core/common/global_flags.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/npu/npu_onerec_block_layer_impl.h"
#include "core/util/env_var.h"
#include "core/util/rec_model_utils.h"

namespace xllm {

inline torch::Tensor pad_encoder_output(const torch::Tensor& encoder_output,
                                        const ModelInputParams& input_params) {
  const auto* onerec_params = input_params.onerec_params();
  CHECK(onerec_params != nullptr) << "OneRec requires onerec_params().";

  const int64_t bs = onerec_params->bs;
  const int64_t hidden_size = encoder_output.size(1);
  const auto& seq_lens = onerec_params->encoder_seq_lens;
  const int64_t max_seq_len = onerec_params->encoder_max_seq_len;

  CHECK_EQ(static_cast<int64_t>(seq_lens.size()), bs)
      << "encoder_seq_lens size mismatch.";

  std::vector<torch::Tensor> seq_list;
  seq_list.reserve(static_cast<size_t>(bs));

  int64_t token_offset = 0;
  for (int64_t i = 0; i < bs; ++i) {
    const int64_t seq_len = seq_lens[i];
    seq_list.emplace_back(encoder_output.narrow(0, token_offset, seq_len));
    token_offset += seq_len;
  }

  auto padded_output = torch::nn::utils::rnn::pad_sequence(
      seq_list, /*batch_first=*/true, /*padding_value=*/0.0);

  if (padded_output.size(1) < max_seq_len) {
    auto extra_padding =
        torch::zeros({bs, max_seq_len - padded_output.size(1), hidden_size},
                     encoder_output.options());
    padded_output = torch::cat({padded_output, extra_padding}, /*dim=*/1);
  }

  return padded_output;
}

inline torch::Tensor compute_onerec_position_bias(
    int64_t query_length,
    int64_t key_length,
    int64_t num_heads,
    bool is_decoder,
    layer::WordEmbedding& position_bias_embedding,
    int64_t num_buckets = 32,
    int64_t max_distance = 128,
    const torch::TensorOptions& options = torch::kFloat32,
    bool is_decode_stage = false,
    const ModelInputParams* input_params = nullptr) {
  auto device = options.device();
  auto dtype = options.dtype();

  int64_t actual_query_length = is_decode_stage ? key_length : query_length;
  if (actual_query_length <= 0) {
    actual_query_length = 1;
  }
  if (key_length <= 0) {
    key_length = 1;
  }

  auto context_position =
      torch::arange(actual_query_length,
                    torch::dtype(torch::kLong).device(device))
          .unsqueeze(1);
  auto memory_position =
      torch::arange(key_length, torch::dtype(torch::kLong).device(device))
          .unsqueeze(0);
  auto relative_position = memory_position - context_position;

  auto relative_buckets = torch::zeros_like(relative_position);

  if (!is_decoder) {
    num_buckets = num_buckets / 2;
    relative_buckets += (relative_position > 0).to(torch::kLong) * num_buckets;
    relative_position = torch::abs(relative_position);
  } else {
    relative_position =
        -torch::min(relative_position, torch::zeros_like(relative_position));
  }

  const int64_t max_exact = num_buckets / 2;
  auto is_small = relative_position < max_exact;
  auto relative_position_if_large =
      max_exact + (torch::log(relative_position.to(torch::kFloat) / max_exact) /
                   std::log(static_cast<double>(max_distance) / max_exact) *
                   (num_buckets - max_exact))
                      .to(torch::kLong);

  relative_position_if_large =
      torch::min(relative_position_if_large,
                 torch::full_like(relative_position_if_large, num_buckets - 1));

  relative_buckets +=
      torch::where(is_small, relative_position, relative_position_if_large);

  auto original_shape = relative_buckets.sizes();
  auto flattened_buckets = relative_buckets.flatten();
  auto values = position_bias_embedding(flattened_buckets);

  if (values.dim() == 2) {
    CHECK_EQ(values.size(0), flattened_buckets.size(0));
    values =
        values.view({original_shape[0], original_shape[1], values.size(1)});
  } else if (values.dim() == 1) {
    values =
        values.unsqueeze(-1).expand({flattened_buckets.size(0), num_heads});
    values = values.view({original_shape[0], original_shape[1], num_heads});
  } else {
    LOG(FATAL) << "Unexpected OneRec position bias dim: " << values.dim();
  }

  if (values.dim() == 3) {
    values = values.permute({2, 0, 1});
  }

  if (is_decode_stage && input_params != nullptr &&
      !input_params->attention.host.kv_seq_lens.empty()) {
    const int32_t seq_kv_len = input_params->attention.host.kv_seq_lens[0];
    values = values.slice(1, -1, values.size(1)).slice(2, 0, seq_kv_len);
  } else if (is_decode_stage) {
    values = values.slice(1, -1, values.size(1));
  }

  return values.to(dtype);
}

class OneRecStackImpl : public torch::nn::Module {
 public:
  OneRecStackImpl(const ModelContext& context,
                  bool is_decode,
                  layer::WordEmbedding& embed_tokens) {
    const auto& args = context.get_model_args();
    const auto& options = context.get_tensor_options();

    hidden_size_ = args.hidden_size();
    is_decoder_ = is_decode;
    use_absolute_position_embedding_ = args.use_absolute_position_embedding();
    use_moe_ = args.use_moe() && is_decoder_;
    num_experts_per_tok_ = args.num_experts_per_tok();
    relative_attention_num_buckets_ = args.relative_attention_num_buckets();
    relative_attention_max_distance_ = args.relative_attention_max_distance();
    num_heads_ = is_decode ? args.decoder_n_heads() : args.n_heads();

    embed_tokens_ = embed_tokens;
    if (!use_absolute_position_embedding_) {
      position_bias_embedding_ = register_module("position_bias_embedding",
                                                 layer::WordEmbedding(context));
    }

    norm_ = register_module("final_layer_norm", layer::RMSNorm(context));

    blocks_ = register_module("block", torch::nn::ModuleList());
    const uint32_t num_layers =
        is_decode ? args.n_layers() : args.n_encoder_layers();
    layers_.reserve(num_layers);
    for (uint32_t i = 0; i < num_layers; ++i) {
      auto block = layer::NpuOneRecBlockLayer(context, is_decode, i);
      layers_.emplace_back(block);
      blocks_->push_back(block);
    }

    (void)options;
  }

  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params,
                        const torch::Tensor& encoder_output = torch::Tensor()) {
    (void)positions;

    const auto* onerec_params = input_params.onerec_params();
    CHECK(onerec_params != nullptr) << "OneRec requires onerec_params().";

    torch::Tensor h;
    if (onerec_params->is_hybrid_mode && !is_decoder_) {
      h = tokens;
    } else if (onerec_params->decoder_context_embedding.defined()) {
      if (tokens.numel() == 0) {
        h = onerec_params->decoder_context_embedding.reshape(
            {-1, onerec_params->decoder_context_embedding.size(-1)});
      } else {
        h = embed_tokens_(tokens);

        auto context_emb = onerec_params->decoder_context_embedding.clone();
        const int64_t hidden_size = context_emb.size(3);
        const int64_t bs = onerec_params->bs;
        const int64_t group_width = onerec_params->group_width;
        const int64_t context_total_tokens = context_emb.size(2);
        const int64_t token_total_tokens = h.size(0);

        const int64_t bs_group = bs * group_width;
        const int64_t seq_len1 =
            token_total_tokens / std::max<int64_t>(1, bs_group);
        const int64_t seq_len2 = context_total_tokens - seq_len1;

        auto token_embedding_reshaped =
            h.view({bs, group_width, seq_len1, hidden_size});
        context_emb.narrow(2, seq_len2, seq_len1)
            .copy_(token_embedding_reshaped);
        h = context_emb.view({-1, hidden_size});
      }
      if (!h.is_contiguous()) {
        h = h.contiguous();
      }
      if (h.device().type() == torch::DeviceType::PrivateUse1 &&
          at_npu::native::get_npu_format(h) != ACL_FORMAT_ND) {
        h = at_npu::native::npu_format_cast(h, ACL_FORMAT_ND).contiguous();
      }
    } else {
      h = embed_tokens_(tokens);
    }

    torch::Tensor npu_encoder_output = encoder_output;
    if (npu_encoder_output.defined() &&
        npu_encoder_output.device().type() != h.device().type()) {
      npu_encoder_output = npu_encoder_output.to(h.device());
    }

    const bool is_prefill =
        onerec_params->rec_stage == OneRecModelInputParams::RecStage::PREFILL;
    auto [query_length, key_length] = compute_sequence_lengths(
        input_params.meta.q_max_seq_len, is_prefill, input_params);

    ModelInputParams input_params_local = input_params;
    auto& mutable_onerec_params = input_params_local.mutable_onerec_params();
    const auto* onerec_xattn_params = input_params.onerec_xattention_params();

    auto validate_selected_token_idxes_stage = [&](const char* stage_name) {
      if (!is_decoder_ || onerec_xattn_params == nullptr ||
          !util::get_bool_env("XLLM_DEBUG_ONEREC_SELECTED_TOKEN_CPU_CHECK",
                              false) ||
          util::get_bool_env("XLLM_DEBUG_ONEREC_SKIP_SELECTED_TOKEN_CPU_CHECK",
                             false) ||
          !onerec_xattn_params->debug_selected_token_idxes.defined() ||
          onerec_xattn_params->debug_selected_token_idxes_expected.empty()) {
        return;
      }
      auto selected_cpu = onerec_xattn_params->debug_selected_token_idxes.to(
          torch::kCPU, /*non_blocking=*/false);
      auto selected_cpu_i64 = selected_cpu.to(torch::kInt64).contiguous();
      auto expected_cpu = torch::tensor(
          onerec_xattn_params->debug_selected_token_idxes_expected,
          torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
      CHECK(torch::equal(selected_cpu_i64, expected_cpu))
          << "OneRec decoder selected_token_idxes changed after " << stage_name
          << ", actual=" << selected_cpu_i64 << ", expected=" << expected_cpu
          << ", rec_stage="
          << static_cast<int32_t>(onerec_xattn_params->rec_stage)
          << ", is_first_prefill=" << onerec_xattn_params->is_first_prefill;
    };

    const bool is_decode_stage = is_decoder_ && !is_prefill;
    torch::Tensor effective_attn_mask;
    if (use_absolute_position_embedding_) {
      const int64_t batch_size =
          std::max<int64_t>(1, input_params.meta.num_sequences);
      effective_attn_mask =
          create_moe_attention_mask(query_length, h, is_decoder_, batch_size);
    } else {
      effective_attn_mask = compute_position_bias_mask(
          query_length, key_length, h, is_decode_stage, input_params);
    }

    auto preprocessed_attn_mask =
        preprocess_attention_mask(effective_attn_mask, h);

    if (mutable_onerec_params.encoder_seq_lens_tensor.defined()) {
      auto flattened_tensor =
          mutable_onerec_params.encoder_seq_lens_tensor.flatten();
      mutable_onerec_params.encoder_seq_lens_tensor =
          flattened_tensor.to(h.device(), torch::kInt).contiguous();
    }

    torch::Tensor expert_array;
    if (use_moe_) {
      const int64_t input_length = h.size(0);
      expert_array = torch::arange(
          0,
          input_length * num_experts_per_tok_,
          torch::TensorOptions().dtype(torch::kInt32).device(h.device()));
    }

    for (size_t i = 0; i < layers_.size(); ++i) {
      aclrtEvent* event = nullptr;
      std::atomic<bool>* event_flag = nullptr;
      if (input_params.parallel.layer_synchronizer) {
        event = input_params.parallel.layer_synchronizer->get_event(i);
        event_flag =
            input_params.parallel.layer_synchronizer->get_event_flag(i);
      }
      if (!input_params.synchronize_layer(i)) {
        return torch::Tensor();
      }

      KVCache dummy_kv_cache;
      if (is_decoder_) {
        CHECK_LT(i, kv_caches.size())
            << "OneRec decoder layer kv_cache is missing at layer " << i;
      }
      KVCache& kv_cache_ref = is_decoder_ ? kv_caches[i] : dummy_kv_cache;

      layers_[i]->forward(
          h,
          preprocessed_attn_mask,
          kv_cache_ref,
          input_params_local,
          npu_encoder_output.defined() ? &npu_encoder_output : nullptr,
          static_cast<int>(i),
          event,
          event_flag,
          expert_array);

      if (input_params.parallel.layer_synchronizer != nullptr &&
          !input_params.parallel.layer_synchronizer->synchronize_layer(i)) {
        return torch::Tensor();
      }
      validate_selected_token_idxes_stage(
          (std::string("decoder_layer_") + std::to_string(i)).c_str());
    }

    validate_selected_token_idxes_stage("decoder_layer_loop");
    std::optional<torch::Tensor> residual = std::nullopt;
    h = std::get<0>(norm_->forward(h, residual));
    validate_selected_token_idxes_stage("decoder_final_norm");
    return h;
  }

  void load_state_dict(const StateDict& state_dict) {
    auto embed_dict = state_dict.get_dict_with_prefix("embed_tokens.");
    if (embed_dict.size() > 0) {
      embed_tokens_->load_state_dict(embed_dict);
    }

    if (!use_absolute_position_embedding_ && position_bias_embedding_) {
      auto pos_bias_dict = state_dict.get_dict_with_prefix(
          "block.0.layer.0.SelfAttention.relative_attention_bias.");
      if (pos_bias_dict.size() > 0) {
        position_bias_embedding_->load_state_dict(pos_bias_dict);
      }
    }

    for (int i = 0; i < static_cast<int>(layers_.size()); ++i) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("block." + std::to_string(i) + "."));
    }

    norm_->load_state_dict(
        state_dict.get_dict_with_prefix("final_layer_norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    for (int i = 0; i < static_cast<int>(layers_.size()); ++i) {
      layers_[i]->verify_loaded_weights(prefix + "block." + std::to_string(i) +
                                        ".");
    }
  }

  void merge_loaded_weights() {
    for (int i = 0; i < static_cast<int>(layers_.size()); ++i) {
      layers_[i]->merge_loaded_weights();
    }
  }

  layer::WordEmbedding get_word_embedding() { return embed_tokens_; }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }

 private:
  std::pair<int64_t, int64_t> compute_sequence_lengths(
      int64_t seq_length,
      bool is_prefill,
      const ModelInputParams& input_params) const {
    int64_t query_length = seq_length;
    int64_t key_length = seq_length;

    const auto* onerec_params = input_params.onerec_params();
    CHECK(onerec_params != nullptr) << "OneRec requires onerec_params().";

    if (is_decoder_) {
      if (is_prefill) {
        query_length = seq_length;
        key_length = seq_length;
      } else {
        query_length = 1;
        if (!input_params.attention.host.kv_seq_lens.empty()) {
          key_length =
              *std::max_element(input_params.attention.host.kv_seq_lens.begin(),
                                input_params.attention.host.kv_seq_lens.end());
        }
        // Decode keeps a square bias/mask shape expected by OneRec NPU block.
        query_length = key_length;
      }
    } else {
      query_length = onerec_params->encoder_max_seq_len;
      key_length = onerec_params->encoder_max_seq_len;
    }

    return {query_length, key_length};
  }

  torch::Tensor create_moe_attention_mask(int64_t seq_length,
                                          const torch::Tensor& h,
                                          bool is_decoder,
                                          int64_t batch_size) const {
    if (!is_decoder) {
      return torch::ones({num_heads_, seq_length, seq_length}, h.options());
    }

    if (use_legacy_onerec_prefill_only_contract()) {
      batch_size = std::max<int64_t>(1, batch_size);
      return torch::triu(torch::ones({seq_length, seq_length},
                                     h.options().dtype(torch::kUInt8)),
                         /*diagonal=*/1)
          .unsqueeze(0)
          .unsqueeze(0)
          .expand({batch_size, 1, seq_length, seq_length})
          .contiguous();
    }

    const float mask_value = -9984.0f;
    torch::Tensor upper_tri_mask =
        torch::triu(torch::ones({seq_length, seq_length},
                                torch::dtype(h.dtype()).device(h.device())),
                    /*diagonal=*/1);
    torch::Tensor expanded_mask = upper_tri_mask.unsqueeze(0).expand(
        {num_heads_, seq_length, seq_length});
    torch::Tensor effective_attn_mask =
        torch::zeros({num_heads_, seq_length, seq_length},
                     torch::dtype(h.dtype()).device(h.device()));
    effective_attn_mask.masked_fill_(expanded_mask.to(torch::kBool),
                                     mask_value);
    return effective_attn_mask;
  }

  torch::Tensor compute_position_bias_mask(
      int64_t query_length,
      int64_t key_length,
      const torch::Tensor& h,
      bool is_decode_stage,
      const ModelInputParams& input_params) {
    CHECK(!position_bias_embedding_.is_empty())
        << "position_bias_embedding is required for relative attention.";

    auto layer_position_bias =
        compute_onerec_position_bias(query_length,
                                     key_length,
                                     num_heads_,
                                     is_decoder_,
                                     position_bias_embedding_,
                                     relative_attention_num_buckets_,
                                     relative_attention_max_distance_,
                                     torch::dtype(h.dtype()).device(h.device()),
                                     is_decode_stage,
                                     &input_params);

    auto effective_attn_mask = layer_position_bias.is_contiguous()
                                   ? layer_position_bias
                                   : layer_position_bias.contiguous();

    if (is_decoder_ && use_legacy_onerec_prefill_only_contract()) {
      const float mask_value = -9984.0f;
      auto upper_tri_mask =
          torch::triu(torch::ones({query_length, query_length},
                                  effective_attn_mask.options()),
                      1);
      auto expanded_mask = upper_tri_mask.unsqueeze(0).expand(
          {num_heads_, query_length, query_length});
      effective_attn_mask.masked_fill_(expanded_mask.to(torch::kBool),
                                       mask_value);
    }

    return effective_attn_mask;
  }

  torch::Tensor preprocess_attention_mask(
      const torch::Tensor& effective_attn_mask,
      const torch::Tensor& h) const {
    if (!effective_attn_mask.defined()) {
      return torch::Tensor();
    }

    auto result = effective_attn_mask;
    if (result.device() != h.device()) {
      result = result.to(h.device());
    }
    if (!result.is_contiguous()) {
      result = result.contiguous();
    }
    if (result.device().type() == torch::DeviceType::PrivateUse1 &&
        at_npu::native::get_npu_format(result) != ACL_FORMAT_ND) {
      result =
          at_npu::native::npu_format_cast(result, ACL_FORMAT_ND).contiguous();
    }
    return result;
  }

  int64_t hidden_size_ = 0;
  bool is_decoder_ = true;
  bool use_absolute_position_embedding_ = false;
  bool use_moe_ = false;
  int64_t relative_attention_num_buckets_ = 32;
  int64_t relative_attention_max_distance_ = 128;
  int64_t num_heads_ = 4;
  int32_t num_experts_per_tok_ = 2;

  layer::WordEmbedding embed_tokens_{nullptr};
  layer::WordEmbedding position_bias_embedding_{nullptr};
  layer::RMSNorm norm_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  std::vector<layer::NpuOneRecBlockLayer> layers_;
};
TORCH_MODULE(OneRecStack);

}  // namespace xllm
