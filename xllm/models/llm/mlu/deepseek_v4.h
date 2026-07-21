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

#include <absl/strings/str_join.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/state_dict/utils.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/deepseek_v4_rotary_embedding.h"
#include "core/layers/common/dsa_metadata.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/mlu/deepseek_v4/deepseek_v4_decoder_layer.h"
#include "core/layers/mlu/deepseek_v4/dsa_cache_mapping.h"
#include "core/layers/mlu/deepseek_v4/dsa_empty_dp_input.h"
#include "core/layers/mlu/deepseek_v4/dsa_metadata_builder_mlu.h"
#include "core/layers/mlu/deepseek_v4/hyper_connection.h"
#include "models/llm/llm_model_base.h"
#include "models/llm/mlu/deepseek_v4_base.h"

namespace xllm::mlu::model {

struct DeepseekV4GraphMetadataState : ModelGraphMetadataState {
  struct DSAMetadataPersistent {
    torch::Tensor attn_mask;
    // MLU-only canonical seq lengths
    torch::Tensor q_cu_seq_lens;
    torch::Tensor kv_cu_seq_lens;
    torch::Tensor q_seq_lens;
    torch::Tensor kv_seq_lens;
    torch::Tensor index_c4_seq_lens;
    // C128 attention
    torch::Tensor c128_context_lens;
    torch::Tensor c128_block_table_for_attn;
    // Sequence lengths
    torch::Tensor seq_lens;
    // Positions
    torch::Tensor input_positions;
    torch::Tensor c4_pad_positions;
    torch::Tensor c128_pad_positions;
    // Per-cache-type block tables and slot mappings — persisted by group_id
    // so graph replay reads from stable device addresses. Multiple layers
    // with the same cache type share the same persistent buffer.
    std::unordered_map<int32_t, torch::Tensor> block_tables_by_group;
    std::unordered_map<int32_t, torch::Tensor> slot_mappings_by_group;
  };
  DSAMetadataPersistent dsa_metadata_persistent;
};

class DeepseekV4ModelImpl final
    : public LlmModelImplBase<layer::DeepseekV4DecoderLayer>,
      private DeepseekV4Base {
 public:
  explicit DeepseekV4ModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::DeepseekV4DecoderLayer>(
            "deepseek_v4",
            context.get_model_args()) {
    const ModelArgs& model_args = context.get_model_args();
    const torch::TensorOptions options = context.get_tensor_options();
    const ParallelArgs& parallel_args = context.get_parallel_args();

    layers_.reserve(model_args.n_layers());
    norm_ = register_module("norm", layer::RMSNorm(context));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));

    hc_mult_ = model_args.hc_mult();
    window_size_ = model_args.window_size();

    num_heads_ = model_args.n_heads();
    dp_local_tp_size_ =
        std::max<int64_t>(parallel_args.world_size() /
                              std::max<int64_t>(parallel_args.dp_size(), 1),
                          1);
    CHECK_EQ(num_heads_ % dp_local_tp_size_, 0)
        << "[DSV4][Init] n_heads must be divisible by local tp size. n_heads="
        << num_heads_ << ", local_tp_size=" << dp_local_tp_size_;

    hc_head_ = register_module(
        "hc_head",
        layer::DeepseekV4HCHead(hc_mult_,
                                model_args.hidden_size(),
                                static_cast<double>(model_args.hc_eps()),
                                static_cast<double>(model_args.rms_norm_eps()),
                                options));

    init_rope(model_args, options);
    init_hadamard(model_args, options);
    max_position_embeddings_ = model_args.max_position_embeddings();

    for (int32_t layer_id = 0; layer_id < model_args.n_layers(); ++layer_id) {
      layers_.emplace_back(layer::DeepseekV4DecoderLayer(context, layer_id));
    }

    build_dsa_cache_info(model_args);

    for (int32_t layer_id = 0; layer_id < model_args.n_layers(); ++layer_id) {
      layers_[static_cast<size_t>(layer_id)]->set_cache_mapping(
          cache_mappings_[static_cast<size_t>(layer_id)]);
    }
  }

  void load_state_dict(const StateDict& state_dict) override {
    embed_tokens_->load_state_dict(state_dict.get_dict_with_prefix(
        std::vector<std::string>{"embed_tokens.", "embed."}));
    for (size_t layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      layers_[layer_id]->load_state_dict(state_dict.get_dict_with_prefix(
          "layers." + std::to_string(layer_id) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
    hc_head_->load_state_dict(state_dict);
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) override {
    torch::NoGradGuard no_grad;
    const bool is_empty_dp_rank = input_params.meta.q_max_seq_len == 0 ||
                                  input_params.meta.num_sequences == 0 ||
                                  tokens.numel() == 0;
    if (tokens.numel() == 0) {
      tokens = torch::tensor(
          {1},
          torch::TensorOptions().dtype(torch::kInt32).device(tokens.device()));
      positions = torch::tensor(
          {0},
          torch::TensorOptions().dtype(torch::kInt32).device(tokens.device()));
    }

    torch::Tensor h = input_params.embedding.input_embedding.defined()
                          ? input_params.embedding.input_embedding
                          : embed_tokens_(tokens);
    if (h.dim() == 2) {
      h = h.unsqueeze(1).repeat({1, hc_mult_, 1});
    }

    const torch::Device runtime_device = h.device();
    tokens = maybe_to_device(tokens, runtime_device);
    positions = maybe_to_device(positions, runtime_device);

    const bool mlu_graph_forward = deepseek_v4_uses_mlu_graph(input_params);

    ModelInputParams modified_input_params = input_params;
    if (is_empty_dp_rank && !mlu_graph_forward) {
      layer::fill_dsv4_empty_dp_params(
          modified_input_params, group_infos_, window_size_);
    }
    std::vector<int32_t>& dp_token_nums =
        modified_input_params.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);

    if (!modified_input_params.attn_metadata ||
        !modified_input_params.attn_metadata->dsa_metadata) {
      CHECK(!mlu_graph_forward)
          << "DeepSeek V4 MLU graph requires prebuilt DSA metadata";
      modified_input_params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              layer::DSAMetadataBuilderMlu::build(modified_input_params,
                                                  positions,
                                                  caches_info_,
                                                  group_infos_,
                                                  window_size_));
    }
    layer::AttentionMetadata& attn_metadata =
        *(modified_input_params.attn_metadata);
    if (is_empty_dp_rank && !mlu_graph_forward) {
      // Empty-DP inputs only preserve local shape and collective participation.
      // They must not write dummy KV rows into real cache slots.
      attn_metadata.is_dummy = true;
    }

    if (!mlu_graph_forward) {
      prepare_dsa_metadata(attn_metadata, runtime_device);
    }

    std::optional<torch::Tensor> residual;
    for (size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
      prepare_layer_metadata(attn_metadata, static_cast<int32_t>(layer_idx));
      h = layers_[layer_idx](h,
                             residual,
                             positions,
                             attn_metadata,
                             kv_caches[layer_idx],
                             modified_input_params,
                             tokens);
      if (!modified_input_params.record_layer(static_cast<uint32_t>(layer_idx),
                                              h.device())) {
        return ModelOutput();
      }
    }

    // Stash the pre-hc_head 3D hidden for the MTP draft.
    torch::Tensor pre_hc_head_h = h;
    h = hc_head_(h);
    auto [hidden_states, residual_out] = norm_(h, std::nullopt);
    ModelOutput out(hidden_states, residual_out);
    if (model_args_.num_speculative_tokens() > 0) {
      out.aux_hidden_states = pre_hc_head_h.flatten(1);
    }
    return out;
  }

  // Re-export base class graph metadata interface as public.
  using DeepseekV4Base::create_graph_forward_metadata_state;
  using DeepseekV4Base::prepare_graph_forward_metadata;
  using DeepseekV4Base::requires_graph_forward_metadata;

 private:
  layer::DeepseekV4HCHead hc_head_{nullptr};
  int64_t num_heads_ = 0;
  int64_t dp_local_tp_size_ = 1;
};
TORCH_MODULE(DeepseekV4Model);

// ============================================================================
// Out-of-line definitions of DeepseekV4Base graph metadata methods.
// These are placed here (not in the base class header) because they need the
// complete DeepseekV4GraphMetadataState type defined above.
// ============================================================================

std::unique_ptr<ModelGraphMetadataState>
DeepseekV4Base::create_graph_forward_metadata_state() {
  return std::make_unique<DeepseekV4GraphMetadataState>();
}

void DeepseekV4Base::prepare_graph_forward_metadata(
    ModelGraphMetadataState* state,
    const torch::Tensor& positions,
    ModelInputParams& input_params) {
  CHECK(state != nullptr)
      << "DeepSeek V4 MLU graph metadata state must be initialized";
  auto* dsv4_state = dynamic_cast<DeepseekV4GraphMetadataState*>(state);
  CHECK(dsv4_state != nullptr)
      << "DeepSeek V4 MLU received incompatible graph metadata state";

  auto modified_input_params = input_params;
  auto& dp_token_nums = modified_input_params.parallel.dp_global_token_nums;
  std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);

  // Build DSA metadata outside graph capture.
  auto attn_metadata = std::make_shared<layer::AttentionMetadata>(
      layer::DSAMetadataBuilderMlu::build(modified_input_params,
                                          positions,
                                          caches_info_,
                                          group_infos_,
                                          window_size_));
  if (!attn_metadata->dsa_metadata) {
    input_params.attn_metadata = attn_metadata;
    return;
  }

  const torch::Device runtime_device =
      positions.defined() ? positions.device() : torch::Device(torch::kCPU);

  prepare_dsa_metadata(*attn_metadata, runtime_device);
  auto& dsa = *attn_metadata->dsa_metadata;
  auto& persistent = dsv4_state->dsa_metadata_persistent;
  init_persistent_cache_buffers(
      /*persistent=*/persistent,
      /*input_params=*/modified_input_params,
      /*num_tokens=*/positions.numel(),
      /*runtime_device=*/runtime_device);
  persist_dsa_metadata(dsa, persistent);
  sync_dsa_seq_metadata(*attn_metadata, dsa);
  input_params.attn_metadata = attn_metadata;
}

class DeepseekV4ForCausalLMImpl final
    : public LlmForCausalLMImplBase<DeepseekV4Model> {
 public:
  explicit DeepseekV4ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekV4Model>(context) {}

  bool requires_graph_forward_metadata() {
    return this->model_->requires_graph_forward_metadata();
  }

  std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return this->model_->create_graph_forward_metadata_state();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& input_params) {
    this->model_->prepare_graph_forward_metadata(
        state, positions, input_params);
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    for (const std::unique_ptr<StateDict>& state_dict :
         loader->get_state_dicts()) {
      std::unordered_map<std::string, torch::Tensor> remapped_dict;
      std::unordered_map<std::string, torch::Tensor> lm_head_dict;
      for (auto it = state_dict->begin(); it != state_dict->end(); ++it) {
        remapped_dict[detail::normalize_model_parameter_name(
            it->first, prefix)] = it->second;
        std::optional<std::string> lm_head_name =
            detail::normalize_lm_head_parameter_name(it->first);
        if (lm_head_name.has_value()) {
          lm_head_dict[lm_head_name.value()] = it->second;
        }
      }

      StateDict remapped_state_dict(remapped_dict);
      model_->load_state_dict(remapped_state_dict);
      if (!lm_head_dict.empty()) {
        lm_head_->load_state_dict(StateDict(lm_head_dict));
      } else {
        lm_head_->load_state_dict(
            remapped_state_dict.get_dict_with_prefix("lm_head."));
      }
    }
  }
};
TORCH_MODULE(DeepseekV4ForCausalLM);

inline void load_deepseek_v4_model_args(const JsonReader& json,
                                        ModelArgs* args) {
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v4");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR_FUNC(hidden_size, "dim", [&] { return args->hidden_size(); });
  LOAD_ARG_OR_FUNC(
      hidden_size, "hidden_size", [&] { return args->hidden_size(); });
  LOAD_ARG_OR_FUNC(
      n_layers, "num_hidden_layers", [&] { return args->n_layers(); });
  LOAD_ARG_OR_FUNC(n_heads, "n_heads", [&] { return args->n_heads(); });
  LOAD_ARG_OR_FUNC(
      n_heads, "num_attention_heads", [&] { return args->n_heads(); });
  LOAD_ARG_OR(o_lora_rank, "o_lora_rank", 1024);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 1024);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 1);
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    if (args->head_dim() > 0) {
      return args->head_dim();
    }
    if (args->hidden_size() > 0 && args->n_heads() > 0) {
      return args->hidden_size() / args->n_heads();
    }
    return int64_t{0};
  });
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR_FUNC(intermediate_size, "intermediate_size", [&] {
    if (args->intermediate_size() > 0) {
      return args->intermediate_size();
    }
    if (args->moe_intermediate_size() > 0) {
      return static_cast<int64_t>(args->moe_intermediate_size());
    }
    if (args->hidden_size() > 0) {
      return args->hidden_size() * 4;
    }
    return int64_t{0};
  });

  LOAD_ARG_OR_FUNC(rms_norm_eps, "rms_norm_eps", [&] {
    return json.value_or<float>("norm_eps", 1e-6f);
  });
  LOAD_ARG_OR_FUNC(
      rope_theta, "rope_theta", [&] { return args->rope_theta(); });
  LOAD_ARG_OR(rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(o_groups, "o_groups", 8);

  LOAD_ARG(compress_ratios, "compress_ratios");
  LOAD_ARG_OR(compress_rope_theta, "compress_rope_theta", 160000.0f);
  LOAD_ARG_OR(window_size, "window_size", 128);

  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 256);
  LOAD_ARG_OR(n_activated_experts, "n_activated_experts", 6);
  LOAD_ARG_OR_FUNC(num_experts_per_tok, "num_experts_per_tok", [&] {
    return args->n_activated_experts();
  });
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 2048);
  LOAD_ARG_OR(swiglu_limit, "swiglu_limit", 10);
  LOAD_ARG_OR(n_hash_layers, "num_hash_layers", 3);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 1.5f);
  LOAD_ARG_OR(scoring_func, "scoring_func", "sqrtsoftplus");

  LOAD_ARG_OR(index_head_dim, "index_head_dim", 128);
  LOAD_ARG_OR(index_n_heads, "index_n_heads", 64);
  LOAD_ARG_OR(index_topk, "index_topk", 512);

  LOAD_ARG_OR(hc_mult, "hc_mult", 4);
  LOAD_ARG_OR(hc_sinkhorn_iters, "hc_sinkhorn_iters", 20);
  LOAD_ARG_OR(hc_eps, "hc_eps", 1e-6f);
  LOAD_ARG_OR(factor, "rope_scaling.factor", 16.0f);
  LOAD_ARG_OR(beta_fast, "rope_scaling.beta_fast", 32.0f);
  LOAD_ARG_OR(beta_slow, "rope_scaling.beta_slow", 1.0f);
  LOAD_ARG_OR(rope_scaling_attn_factor, "rope_scaling.attn_factor", 1.0f);
  LOAD_ARG_OR(scale_fmt, "scale_fmt", "ue8m0");

  LOAD_ARG_OR_FUNC(
      max_batch_size, "max_batch_size", [&] { return args->max_batch_size(); });
  LOAD_ARG_OR_FUNC(
      max_seq_len, "max_seq_len", [&] { return args->max_seq_len(); });
  LOAD_ARG_OR_FUNC(
      vocab_size, "vocab_size", [&] { return args->vocab_size(); });
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 163840);

  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
}

struct DeepseekV4ArgsPolicy {
  std::unordered_set<int32_t> supported_compress_ratios;
  std::unordered_set<int32_t> supported_effective_ratios;
  std::unordered_set<std::string> supported_score_funcs;
  int32_t default_compress_ratio = 1;
};

inline DeepseekV4ArgsPolicy build_deepseek_v4_args_policy() {
  DeepseekV4ArgsPolicy policy;
  policy.supported_compress_ratios = {0, 1, 4, 128};
  policy.supported_effective_ratios = {1, 4, 128};
  policy.supported_score_funcs = {"softmax", "sigmoid", "sqrtsoftplus"};
  policy.default_compress_ratio = 1;
  return policy;
}

inline void process_deepseek_v4_args(ModelArgs* args,
                                     const DeepseekV4ArgsPolicy& policy) {
  SET_ARG(n_activated_experts, args->num_experts_per_tok());
  if (args->n_layers() > 0 &&
      static_cast<int64_t>(args->compress_ratios().size()) < args->n_layers()) {
    args->compress_ratios().resize(static_cast<size_t>(args->n_layers()),
                                   policy.default_compress_ratio);
  }
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
}

inline void validate_deepseek_v4_args(const ModelArgs& args,
                                      const DeepseekV4ArgsPolicy& policy) {
  CHECK_GT(args.n_layers(), 0)
      << "deepseek_v4 config n_layers/num_hidden_layers must be > 0, got "
      << args.n_layers();
  CHECK_GE(static_cast<int64_t>(args.compress_ratios().size()), args.n_layers())
      << "deepseek_v4 config compress_ratios size must be >= n_layers after "
         "processing, got "
      << args.compress_ratios().size() << " vs " << args.n_layers();
  for (int64_t layer_id = 0; layer_id < args.n_layers(); ++layer_id) {
    const int32_t ratio = args.compress_ratios()[static_cast<size_t>(layer_id)];
    CHECK(policy.supported_compress_ratios.count(ratio) > 0)
        << "deepseek_v4 config compress_ratios[" << layer_id
        << "] must be in supported_compress_ratios, got " << ratio;
    const int32_t effective_ratio = normalize_compress_ratio(ratio);
    CHECK(policy.supported_effective_ratios.count(effective_ratio) > 0)
        << "deepseek_v4 config effective compress_ratios[" << layer_id
        << "] must be one of 1/4/128, got " << effective_ratio;
  }
  CHECK_GT(args.window_size(), 0)
      << "deepseek_v4 config window_size must be > 0, got "
      << args.window_size();
  CHECK_GT(args.n_routed_experts(), 0)
      << "deepseek_v4 config n_routed_experts must be > 0, got "
      << args.n_routed_experts();
  CHECK_GT(args.n_activated_experts(), 0)
      << "deepseek_v4 config n_activated_experts/num_experts_per_tok must be "
         "> 0, got "
      << args.n_activated_experts();
  CHECK_LE(args.n_activated_experts(), args.n_routed_experts())
      << "deepseek_v4 config n_activated_experts/num_experts_per_tok must be "
         "<= n_routed_experts, got "
      << args.n_activated_experts() << " vs " << args.n_routed_experts();
  CHECK_GE(args.n_hash_layers(), 0)
      << "deepseek_v4 config num_hash_layers/n_hash_layers must be >= 0, got "
      << args.n_hash_layers();
  CHECK_GT(args.routed_scaling_factor(), 0.0f)
      << "deepseek_v4 config routed_scaling_factor/route_scale must be > 0, "
         "got "
      << args.routed_scaling_factor();
  CHECK(!args.scoring_func().empty())
      << "deepseek_v4 config scoring_func/score_func must not be empty";

  std::string score_func = args.scoring_func();
  std::transform(
      score_func.begin(),
      score_func.end(),
      score_func.begin(),
      [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  CHECK(policy.supported_score_funcs.count(score_func) > 0)
      << "deepseek_v4 config scoring_func/score_func must be in "
      << absl::StrJoin(policy.supported_score_funcs, ", ") << ", got "
      << args.scoring_func();

  CHECK_GT(args.index_head_dim(), 0)
      << "deepseek_v4 config index_head_dim must be > 0, got "
      << args.index_head_dim();
  CHECK_GT(args.index_n_heads(), 0)
      << "deepseek_v4 config index_n_heads must be > 0, got "
      << args.index_n_heads();
  CHECK_GT(args.index_topk(), 0)
      << "deepseek_v4 config index_topk must be > 0, got " << args.index_topk();
  CHECK_GT(args.hc_mult(), 0)
      << "deepseek_v4 config hc_mult must be > 0, got " << args.hc_mult();
  CHECK_GE(args.hc_sinkhorn_iters(), 0)
      << "deepseek_v4 config hc_sinkhorn_iters must be >= 0, got "
      << args.hc_sinkhorn_iters();
  CHECK_GT(args.hc_eps(), 0.0f)
      << "deepseek_v4 config hc_eps must be > 0, got " << args.hc_eps();
  CHECK_GT(args.factor(), 0.0f)
      << "deepseek_v4 requires positive rope_scaling.factor/factor, got "
      << args.factor();
  CHECK_GT(args.rope_scaling_attn_factor(), 0.0f)
      << "deepseek_v4 requires positive rope_scaling_attn_factor, got "
      << args.rope_scaling_attn_factor();
  CHECK_GT(args.rope_theta(), 0.0f)
      << "deepseek_v4 requires positive rope_theta, got " << args.rope_theta();
  CHECK_GT(args.compress_rope_theta(), 0.0f)
      << "deepseek_v4 requires positive compress_rope_theta, got "
      << args.compress_rope_theta();
}

REGISTER_CAUSAL_MODEL(deepseek_v4, DeepseekV4ForCausalLM);

REGISTER_MODEL_ARGS(deepseek_v4, [&] {
  const DeepseekV4ArgsPolicy args_policy = build_deepseek_v4_args_policy();
  load_deepseek_v4_model_args(json, args);
  process_deepseek_v4_args(args, args_policy);
  validate_deepseek_v4_args(*args, args_policy);
});

}  // namespace xllm::mlu::model
