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

#include <glog/logging.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "core/common/global_flags.h"
#include "core/common/interruption_bus.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/model_loader.h"
#include "core/layers/common/attention_mask.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/common/qwen3_next_rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/npu_torch/minimax_m3_decode_layer.h"

namespace xllm::npu::model {

class MiniMaxM3ModelImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxM3ModelImpl(const ModelContext& context) {
    InterruptionBus::get_instance().subscribe(
        [this](bool interrupted) { layer_forward_interrupted_ = interrupted; });

    const torch::TensorOptions& options = context.get_tensor_options();
    const ModelArgs& model_args = context.get_model_args();
    const ParallelArgs& parallel_args = context.get_parallel_args();

    hidden_size_ = model_args.hidden_size();
    enable_mla_ = model_args.enable_mla();

    embed_tokens_ =
        register_module("embed_tokens",
                        layer::WordEmbedding(model_args.vocab_size(),
                                             model_args.hidden_size(),
                                             parallel_args,
                                             options));
    norm_ = register_module(
        "norm",
        layer::Qwen3NextRMSNorm(
            model_args.hidden_size(), model_args.rms_norm_eps(), options));

    const int32_t mask_value = FLAGS_enable_chunked_prefill ? -9984 : 1;
    attn_mask_ = layer::AttentionMask(
        options.device(), options.dtype().toScalarType(), mask_value);

    layers_.reserve(model_args.n_layers());
    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      layers_.push_back(register_module(
          std::to_string(i), layer::MiniMaxM3DecoderLayer(context, i)));
    }
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    ModelInputParams modified_input_params = input_params;
    torch::Tensor h;
    if (input_params.embedding.input_embedding.defined()) {
      h = input_params.embedding.input_embedding;
    } else if (tokens.numel() == 0) {
      h = torch::empty({0, hidden_size_}, embed_tokens_->weight().options());
    } else {
      h = embed_tokens_(tokens);
    }

    if (!modified_input_params.attn_metadata) {
      modified_input_params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              get_attention_metadata(modified_input_params, h));
    }

    layer::AttentionMetadata& attn_metadata =
        *(modified_input_params.attn_metadata);
    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < layers_.size(); ++i) {
      h = layers_[i]->forward(h,
                              residual,
                              positions,
                              attn_metadata,
                              kv_caches[i],
                              modified_input_params);
    }

    if (h.numel() == 0) {
      return ModelOutput(h);
    }

    std::optional<torch::Tensor> residual_out;
    std::tie(h, residual_out) = norm_(h, residual);
    return ModelOutput(h, residual_out);
  }

  void load_state_dict(const StateDict& state_dict) {
    embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));

    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }

    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return embed_tokens_(input_ids);
  }

  layer::WordEmbedding get_word_embedding() { return embed_tokens_; }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }

 private:
  layer::AttentionMetadata get_attention_metadata(
      const ModelInputParams& params,
      const torch::Tensor& h) {
    if (params.meta.q_max_seq_len == 0) {
      return layer::AttentionMetadataBuilder::build(params, enable_mla_);
    }

    max_seq_len_ = std::max(params.meta.kv_max_seq_len, max_seq_len_);
    torch::Tensor attn_mask;
    if (FLAGS_enable_chunked_prefill) {
      const int32_t max_kv_seq = params.meta.kv_max_seq_len;
      const int32_t num_sequences = params.meta.num_sequences;
      if (num_sequences > 0) {
        std::vector<torch::Tensor> req_mask_vec;
        req_mask_vec.reserve(num_sequences);
        for (int32_t i = 0; i < num_sequences; ++i) {
          req_mask_vec.emplace_back(
              attn_mask_.gen_append_mask(params.attention.host.q_seq_lens[i],
                                         params.attention.host.kv_seq_lens[i],
                                         max_kv_seq,
                                         h.dtype().toScalarType(),
                                         h.device()));
        }
        attn_mask = torch::cat(req_mask_vec, 0);
      } else {
        attn_mask = attn_mask_.get_attn_mask(
            max_seq_len_, h.dtype().toScalarType(), h.device());
      }
    } else {
      attn_mask = attn_mask_.get_attn_mask(
          max_seq_len_, h.dtype().toScalarType(), h.device());
    }
    return layer::AttentionMetadataBuilder::build(
        params, enable_mla_, attn_mask);
  }

  std::vector<layer::MiniMaxM3DecoderLayer> layers_;
  layer::WordEmbedding embed_tokens_{nullptr};
  layer::Qwen3NextRMSNorm norm_{nullptr};
  layer::AttentionMask attn_mask_;
  int64_t hidden_size_ = 0;
  int32_t max_seq_len_ = 0;
  bool enable_mla_ = false;
  bool layer_forward_interrupted_ = false;
};
TORCH_MODULE(MiniMaxM3Model);

class MiniMaxM3ForCausalLMImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxM3ForCausalLMImpl(const ModelContext& context) {
    tie_word_embeddings_ = context.get_model_args().tie_word_embeddings();
    model_ = register_module("model", MiniMaxM3Model(context));
    lm_head_ = register_module("lm_head", layer::LmHead(context));
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    return model_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) {
    torch::Tensor h = hidden_states;
    if (selected_idxes.defined()) {
      h = h.index_select(/*dim=*/0, selected_idxes);
    }
    return lm_head_(h);
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) {
    if (selected_idxes.defined()) {
      return hidden_states.index_select(/*dim=*/0, selected_idxes);
    }
    return hidden_states;
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "language_model.model.") {
    for (const std::unique_ptr<StateDict>& state_dict :
         loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix(prefix));
      if (tie_word_embeddings_) {
        lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix(prefix + "embed_tokens."));
      } else {
        lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix("language_model.lm_head."));
      }
    }
  }

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) {
    (void)layer_id;
    (void)expert_ids;
  }

  void update_expert_weight(int32_t layer_id) { (void)layer_id; }

  layer::LmHead get_lm_head() { return lm_head_; }

  void set_lm_head(layer::LmHead& head) { lm_head_ = head; }

  layer::WordEmbedding get_word_embedding() {
    return model_->get_word_embedding();
  }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    model_->set_word_embedding(word_embedding);
  }

 private:
  MiniMaxM3Model model_{nullptr};
  bool tie_word_embeddings_ = false;
  layer::LmHead lm_head_{nullptr};
};
TORCH_MODULE(MiniMaxM3ForCausalLM);

REGISTER_CAUSAL_MODEL(minimax_m3, MiniMaxM3ForCausalLM);
REGISTER_CAUSAL_MODEL(minimax_m3_vl, MiniMaxM3ForCausalLM);

REGISTER_MODEL_ARGS(minimax_m3_vl, [&] {
  LOAD_ARG_OR(model_type, "model_type", "minimax_m3_vl");
  LOAD_ARG_OR(dtype, "text_config.dtype", "bfloat16");
  LOAD_ARG_OR(attention_bias, "text_config.attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "text_config.attention_dropout", 0.0f);
  LOAD_ARG_OR(bos_token_id, "text_config.bos_token_id", 200019);
  LOAD_ARG_OR(eos_token_id, "text_config.eos_token_id", 200020);
  LOAD_ARG_OR(pad_token_id, "text_config.pad_token_id", 200000);
  LOAD_ARG_OR(vision_start_token_id, "vision_start_token_id", 200029);
  LOAD_ARG_OR(vision_end_token_id, "vision_end_token_id", 200030);
  LOAD_ARG_OR(image_token_id, "image_token_index", 200025);
  LOAD_ARG_OR(video_token_id, "video_token_index", 200026);
  LOAD_ARG_OR(head_dim, "text_config.head_dim", 128);
  LOAD_ARG_OR(rotary_dim, "text_config.rotary_dim", 64);
  LOAD_ARG_OR(hidden_act, "text_config.hidden_act", "swigluoai");
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 6144);
  LOAD_ARG_OR(intermediate_size, "text_config.dense_intermediate_size", 12288);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 524288);
  LOAD_ARG_OR(moe_intermediate_size, "text_config.intermediate_size", 3072);
  LOAD_ARG_OR(shared_expert_intermediate_size,
              "text_config.shared_intermediate_size",
              3072);
  LOAD_ARG_OR(n_shared_experts, "text_config.n_shared_experts", 1);
  LOAD_ARG_OR(norm_topk_prob, "text_config.norm_topk_prob", true);
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 64);
  LOAD_ARG_OR(num_experts, "text_config.num_local_experts", 128);
  LOAD_ARG_OR(n_routed_experts, "text_config.num_local_experts", 128);
  LOAD_ARG_OR(num_experts_per_tok, "text_config.num_experts_per_tok", 4);
  LOAD_ARG_OR(n_group, "text_config.n_group", 1);
  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 60);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 4);
  LOAD_ARG_OR(output_router_logits, "text_config.output_router_logits", false);
  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-6);
  LOAD_ARG_OR(rope_theta, "text_config.rope_theta", 5000000.0f);
  LOAD_ARG_OR(scoring_func, "text_config.scoring_func", "sigmoid");
  LOAD_ARG_OR(swiglu_limit, "text_config.swiglu_limit", 7.0f);
  LOAD_ARG_OR(topk_group, "text_config.topk_group", 1);
  LOAD_ARG_OR(routed_scaling_factor, "text_config.routed_scaling_factor", 2.0f);
  LOAD_ARG_OR(router_aux_loss_coef, "text_config.router_aux_loss_coef", 0.0f);
  LOAD_ARG_OR(use_sliding_window, "text_config.use_sliding_window", false);
  LOAD_ARG_OR(tie_word_embeddings, "text_config.tie_word_embeddings", false);
  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 200064);
  LOAD_ARG_OR(
      mlp_only_layers, "text_config.mlp_only_layers", std::vector<int32_t>());

  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "gelu");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1280);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 5120);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_attention_heads", 16);
  LOAD_ARG_OR(mm_num_channels, "vision_config.num_channels", 3);
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.num_hidden_layers", 32);
  LOAD_ARG_OR(mm_image_size, "vision_config.image_size", 2016);
  LOAD_ARG_OR(mm_layer_norm_eps, "vision_config.layer_norm_eps", 1e-5f);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.projection_dim", 6144);
  LOAD_ARG_OR(mm_projector_hidden_size, "projector_hidden_size", 6144);
  LOAD_ARG_OR(mm_projector_hidden_act, "projector_hidden_act", "gelu");
  LOAD_ARG_OR(mm_spatial_merge_size,
              "vision_config.img_token_compression_config.spatial_merge_size",
              2);
  LOAD_ARG_OR(mm_temporal_patch_size,
              "vision_config.img_token_compression_config.temporal_patch_size",
              2);
  LOAD_ARG_OR(mm_image_patch_size, "vision_config.patch_size", 14);
  LOAD_ARG_OR(mm_image_merge_size,
              "vision_config.img_token_compression_config.spatial_merge_size",
              2);
  LOAD_ARG_OR(mm_image_temporal_patch_size,
              "vision_config.img_token_compression_config.temporal_patch_size",
              2);
  LOAD_ARG_OR(mm_image_min_pixels, "min_pixels", 3136);
  LOAD_ARG_OR(mm_image_max_pixels, "max_pixels", 451584);
  SET_ARG(mm_image_normalize_mean,
          std::vector<double>({0.48145466, 0.4578275, 0.40821073}));
  SET_ARG(mm_image_normalize_std,
          std::vector<double>({0.26862954, 0.26130258, 0.27577711}));
  SET_ARG(mm_head_dim, args->mm_hidden_size() / args->mm_num_attention_heads());

  LOAD_ARG_OR(first_k_dense_replace, "text_config.first_k_dense_replace", 3);
  LOAD_ARG_OR(use_sparse_attention,
              "text_config.sparse_attention_config.use_sparse_attention",
              false);
  LOAD_ARG_OR(sparse_index_dim,
              "text_config.sparse_attention_config.sparse_index_dim",
              128);
  LOAD_ARG_OR(sparse_num_index_heads,
              "text_config.sparse_attention_config.sparse_num_index_heads",
              4);
  LOAD_ARG_OR(sparse_topk_blocks,
              "text_config.sparse_attention_config.sparse_topk_blocks",
              16);
  LOAD_ARG_OR(sparse_block_size,
              "text_config.sparse_attention_config.sparse_block_size",
              128);
  LOAD_ARG_OR(sparse_init_block,
              "text_config.sparse_attention_config.sparse_init_block",
              0);
  LOAD_ARG_OR(sparse_local_block,
              "text_config.sparse_attention_config.sparse_local_block",
              1);
  LOAD_ARG_OR(sparse_attention_freq,
              "text_config.sparse_attention_config.sparse_attention_freq",
              std::vector<int32_t>());

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
  SET_ARG(topk_method, "noaux_tc");
  if (args->use_sparse_attention()) {
    SET_ARG(index_head_dim, args->sparse_index_dim());
    SET_ARG(index_n_heads, args->sparse_num_index_heads());
    SET_ARG(index_topk, args->sparse_topk_blocks());
  }
});

REGISTER_MODEL_ARGS_LOADER(minimax_m3,
                           [](const JsonReader& json, ModelArgs* args) {
                             const bool loaded =
                                 ModelRegistry::get_model_args_loader(
                                     "minimax_m3_vl")(json, args);
                             SET_ARG(model_type, "minimax_m3");
                             return loaded;
                           });

}  // namespace xllm::npu::model
