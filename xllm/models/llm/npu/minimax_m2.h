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

#include <absl/strings/match.h>
#include <absl/strings/str_replace.h>
#include <glog/logging.h>

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/interruption_bus.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/hf_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/model_loader.h"
#include "core/layers/common/attention_mask.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/npu_torch/minimax_m2_decode_layer.h"

namespace xllm::npu::model {

class MiniMaxM2MoeDecoderLayerImpl : public torch::nn::Module {
 public:
  MiniMaxM2MoeDecoderLayerImpl(const ModelContext& context, int32_t layer_id) {
    const auto& quant_args = context.get_quant_args();

    if (quant_args.quant_method() == "fp8") {
      enable_fp8_dynamic_dequant_ = true;
      if (quant_args.weight_block_size().size() == 2 &&
          quant_args.weight_block_size()[0] > 0 &&
          quant_args.weight_block_size()[1] > 0) {
        fp8_weight_block_size_ = {quant_args.weight_block_size()[0],
                                  quant_args.weight_block_size()[1]};
      }
    }

    layer_ = register_module("layer",
                             layer::MiniMaxM2DecoderLayer(context, layer_id));
  }

  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const layer::AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params) {
    return layer_->forward(
        x, residual, positions, attn_metadata, kv_cache, input_params);
  }

  void load_state_dict(const StateDict& state_dict) {
    std::unordered_map<std::string, torch::Tensor> remapped;
    remapped.reserve(state_dict.size());
    std::unordered_map<std::string, torch::Tensor> pending_fp8_weights;
    std::unordered_map<std::string, torch::Tensor> pending_fp8_scales;

    for (const auto& [name, tensor] : state_dict) {
      std::string mapped_name = remap_layer_weight_name(name);
      if (enable_fp8_dynamic_dequant_) {
        if (absl::EndsWith(mapped_name, ".weight_scale_inv")) {
          const std::string paired_weight_name =
              mapped_name.substr(0, mapped_name.size() - 10);
          auto pending_weight = pending_fp8_weights.find(paired_weight_name);
          if (pending_weight != pending_fp8_weights.end()) {
            remapped.emplace(
                paired_weight_name,
                dequantize_fp8_block_weight(
                    pending_weight->second, tensor, fp8_weight_block_size_));
            pending_fp8_weights.erase(pending_weight);
          } else {
            pending_fp8_scales.emplace(mapped_name, tensor);
          }
          continue;
        }

        if (absl::EndsWith(mapped_name, ".weight") &&
            is_fp8_dtype(tensor.scalar_type())) {
          const std::string scale_name = mapped_name + "_scale_inv";
          auto pending_scale = pending_fp8_scales.find(scale_name);
          if (pending_scale != pending_fp8_scales.end()) {
            remapped.emplace(
                mapped_name,
                dequantize_fp8_block_weight(
                    tensor, pending_scale->second, fp8_weight_block_size_));
            pending_fp8_scales.erase(pending_scale);
          } else {
            pending_fp8_weights.emplace(mapped_name, tensor);
          }
          continue;
        }
      }

      remapped.emplace(mapped_name, tensor);
    }

    if (enable_fp8_dynamic_dequant_) {
      CHECK(pending_fp8_weights.empty() && pending_fp8_scales.empty())
          << "Unpaired fp8 MiniMax-M2 weight/scale tensors detected: "
          << "pending_weights=" << pending_fp8_weights.size()
          << ", pending_scales=" << pending_fp8_scales.size();
    }

    layer_->load_state_dict(StateDict(std::move(remapped)));
  }

 private:
  static std::string remap_layer_weight_name(const std::string& name) {
    std::string mapped_name = name;
    if (absl::StartsWith(mapped_name, "block_sparse_moe.")) {
      mapped_name =
          absl::StrReplaceAll(mapped_name, {{"block_sparse_moe.", "mlp."}});
    }
    if (mapped_name == "mlp.e_score_correction_bias") {
      return "mlp.gate.e_score_correction_bias";
    }
    mapped_name = absl::StrReplaceAll(mapped_name,
                                      {{".w1.", ".gate_proj."},
                                       {".w2.", ".down_proj."},
                                       {".w3.", ".up_proj."}});
    return mapped_name;
  }

  static bool is_fp8_dtype(torch::ScalarType dtype) {
    return dtype == torch::kFloat8_e5m2 || dtype == torch::kFloat8_e4m3fn;
  }

  static torch::Tensor dequantize_fp8_block_weight(
      const torch::Tensor& fp8_weight,
      const torch::Tensor& weight_scale_inv,
      const std::array<int64_t, 2>& block_size) {
    CHECK_EQ(fp8_weight.dim(), 2)
        << "Only 2D fp8 weights are supported, got shape "
        << fp8_weight.sizes();
    CHECK_EQ(weight_scale_inv.dim(), 2)
        << "FP8 weight scale tensor must be 2D, got shape "
        << weight_scale_inv.sizes();

    const int64_t block_n = block_size[0];
    const int64_t block_k = block_size[1];
    const int64_t n = fp8_weight.size(0);
    const int64_t k = fp8_weight.size(1);
    const int64_t n_tiles = (n + block_n - 1) / block_n;
    const int64_t k_tiles = (k + block_k - 1) / block_k;

    CHECK_EQ(weight_scale_inv.size(0), n_tiles)
        << "Unexpected fp8 scale shape " << weight_scale_inv.sizes()
        << " for weight shape " << fp8_weight.sizes();
    CHECK_EQ(weight_scale_inv.size(1), k_tiles)
        << "Unexpected fp8 scale shape " << weight_scale_inv.sizes()
        << " for weight shape " << fp8_weight.sizes();

    if (n % block_n == 0 && k % block_k == 0) {
      auto weight_bf16 = fp8_weight.to(torch::kBFloat16)
                             .reshape({n_tiles, block_n, k_tiles, block_k});
      auto scale_bf16 = weight_scale_inv.to(torch::kBFloat16)
                            .reshape({n_tiles, 1, k_tiles, 1});
      return (weight_bf16 * scale_bf16).reshape({n, k});
    }

    auto expanded_scale = weight_scale_inv.repeat_interleave(block_n, 0)
                              .repeat_interleave(block_k, 1);
    expanded_scale = expanded_scale.slice(/*dim=*/0, /*start=*/0, /*end=*/n)
                         .slice(/*dim=*/1, /*start=*/0, /*end=*/k)
                         .to(torch::kBFloat16);
    return fp8_weight.to(torch::kBFloat16) * expanded_scale;
  }

  layer::MiniMaxM2DecoderLayer layer_{nullptr};
  bool enable_fp8_dynamic_dequant_ = false;
  std::array<int64_t, 2> fp8_weight_block_size_ = {128, 128};
};
TORCH_MODULE(MiniMaxM2MoeDecoderLayer);

class MiniMaxM2ModelImpl : public torch::nn::Module {
 public:
  explicit MiniMaxM2ModelImpl(const ModelContext& context) {
    InterruptionBus::get_instance().subscribe(
        [this](bool interrupted) { layer_forward_interrupted_ = interrupted; });

    const auto& options = context.get_tensor_options();
    const auto& model_args = context.get_model_args();
    const auto& parallel_args = context.get_parallel_args();

    enable_mla_ = model_args.enable_mla();

    embed_tokens_ =
        register_module("embed_tokens",
                        layer::WordEmbedding(model_args.vocab_size(),
                                             model_args.hidden_size(),
                                             parallel_args,
                                             options));
    norm_ = register_module("norm", layer::RMSNorm(context));

    int32_t mask_value =
        ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() ? -9984
                                                                         : 1;
    attn_mask_ = layer::AttentionMask(
        options.device(), options.dtype().toScalarType(), mask_value);

    layers_.reserve(model_args.n_layers());
    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      layers_.push_back(register_module(std::to_string(i),
                                        MiniMaxM2MoeDecoderLayer(context, i)));
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

    auto& attn_metadata = *(modified_input_params.attn_metadata);
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

    auto [hidden_states, residual_out] = norm_(h, residual);
    return ModelOutput(hidden_states, residual_out);
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
    if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
      const int32_t max_kv_seq = params.meta.kv_max_seq_len;
      const int32_t num_sequences = params.meta.num_sequences;
      if (num_sequences > 0) {
        std::vector<torch::Tensor> req_mask_vec;
        req_mask_vec.reserve(num_sequences);
        for (int32_t j = 0; j < num_sequences; ++j) {
          req_mask_vec.emplace_back(
              attn_mask_.gen_append_mask(params.attention.host.q_seq_lens[j],
                                         params.attention.host.kv_seq_lens[j],
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

  std::vector<MiniMaxM2MoeDecoderLayer> layers_;
  layer::WordEmbedding embed_tokens_{nullptr};
  layer::RMSNorm norm_{nullptr};
  layer::AttentionMask attn_mask_;
  int64_t hidden_size_ = 0;
  int32_t max_seq_len_ = 0;
  bool enable_mla_ = false;
  bool layer_forward_interrupted_ = false;
};
TORCH_MODULE(MiniMaxM2Model);

class MiniMaxM2ForCausalLMImpl : public torch::nn::Module {
 public:
  explicit MiniMaxM2ForCausalLMImpl(const ModelContext& context) {
    tie_word_embeddings_ = context.get_model_args().tie_word_embeddings();
    model_ = register_module("model", MiniMaxM2Model(context));
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
    auto h = hidden_states;
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
                  std::string prefix = "model.") {
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix(prefix));
      if (tie_word_embeddings_) {
        lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix(prefix + "embed_tokens."));
      } else {
        lm_head_->load_state_dict(state_dict->get_dict_with_prefix("lm_head."));
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
  MiniMaxM2Model model_{nullptr};
  bool tie_word_embeddings_ = false;
  layer::LmHead lm_head_{nullptr};
};
TORCH_MODULE(MiniMaxM2ForCausalLM);

REGISTER_CAUSAL_MODEL(minimax_m2, MiniMaxM2ForCausalLM);

REGISTER_MODEL_ARGS(minimax_m2, [&] {
  LOAD_ARG_OR(model_type, "model_type", "minimax_m2");
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(attention_bias, "attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 200019);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 200020);
  LOAD_ARG_OR(head_dim, "head_dim", 128);
  LOAD_ARG_OR(rotary_dim, "rotary_dim", 64);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 3072);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 1536);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 196608);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 62);
  LOAD_ARG_OR(moe_intermediate_size, "intermediate_size", 1536);
  SET_ARG(n_shared_experts, 0);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 48);
  LOAD_ARG_OR(num_experts, "num_local_experts", 256);
  LOAD_ARG_OR(n_routed_experts, "num_local_experts", 256);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(n_group, "n_group", 1);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 62);
  if (json.contains("attn_type_list")) {
    const auto attn_type_list =
        json.value_or<std::vector<int>>("attn_type_list", std::vector<int>());
    if (!attn_type_list.empty() &&
        args->n_layers() != static_cast<int32_t>(attn_type_list.size())) {
      LOG(WARNING) << "MiniMax config mismatch: num_hidden_layers="
                   << args->n_layers()
                   << ", attn_type_list size=" << attn_type_list.size()
                   << ". Using attn_type_list size.";
      args->n_layers() = static_cast<int32_t>(attn_type_list.size());
    }
  }
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 8);
  LOAD_ARG_OR(output_router_logits, "output_router_logits", false);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(rope_theta, "rope_theta", 5000000.0f);
  LOAD_ARG_OR(scoring_func, "scoring_func", "sigmoid");
  LOAD_ARG_OR(topk_group, "topk_group", 1);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 1.0f);
  LOAD_ARG_OR(router_aux_loss_coef, "router_aux_loss_coef", 0.0f);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(vocab_size, "vocab_size", 200064);
  LOAD_ARG_OR(mlp_only_layers, "mlp_only_layers", std::vector<int>());

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
  SET_ARG(topk_method, "noaux_tc");
});

}  // namespace xllm::npu::model
