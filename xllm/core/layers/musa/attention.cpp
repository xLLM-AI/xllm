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

#include "attention.h"

#include <cstdint>
#include <tuple>
#include <vector>

#include "MTTOplib/Attention.h"
#include "MTTOplib/Ops.h"
#include "MTTOplib/WeightReorder.h"

namespace xllm {
namespace layer {
AttentionImpl::AttentionImpl(ModelArgs const& args,
                             QuantArgs const& quant_args,
                             ParallelArgs const& parallel_args,
                             torch::TensorOptions const& options)
    : MUSALayerBaseImpl(options),
      num_heads_(args.n_heads()),
      num_kv_heads_(args.n_kv_heads().value_or(args.n_heads())),
      head_dim_(args.head_dim()),
      q_size_(num_heads_ * head_dim_),
      kv_size_(num_kv_heads_ * head_dim_),
      rms_eps(args.rms_norm_eps()),
      scaling_(std::sqrt(1.0f / head_dim_)),
      hidden_size_(args.hidden_size()) {
  weights_.resize(weight_num_);
}

AttentionImpl::AttentionImpl(int64_t num_heads,
                             int64_t head_size,
                             float scale,
                             int64_t num_kv_heads,
                             int64_t sliding_window) {}

torch::Tensor AttentionImpl::forward(torch::Tensor& input,
                                     ForwardParams& fwd_params) {
  auto&& cache = fwd_params.kv_cache;
  auto& input_params = const_cast<ModelInputParams&>(fwd_params.input_params);

  auto musa_attn_meta = xllm_musa::AttnMetaData::build(
      input_params.attention.host.q_seq_lens,
      input_params.attention.host.kv_seq_lens,
      num_heads_,
      num_kv_heads_,
      head_dim_,
      input_params.attention.device.new_cache_slots,
      64);

  return xllm_musa::QWen3Attn(input,
                              cache.get_k_cache(),
                              cache.get_v_cache(),
                              input_params.attention.device.block_tables,
                              fwd_params.attn_meta.mrope_cos,
                              fwd_params.positions,
                              weights_,
                              rms_eps,
                              musa_attn_meta);
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  // This method is not used in the current implementation
  return std::make_tuple(torch::Tensor(), std::nullopt);
}

void AttentionImpl::load_state_dict(StateDict const& state_dict) {
  using WeightMeta = std::pair<std::string, std::vector<int64_t>>;
  static int32_t all_loaded = 0;
  std::vector<WeightMeta> meta = {{"q_proj.", {q_size_, hidden_size_}},
                                  {"k_proj.", {kv_size_, hidden_size_}},
                                  {"v_proj.", {kv_size_, hidden_size_}},
                                  {"o_proj.", {hidden_size_, hidden_size_}},
                                  {"q_norm.", {128}},
                                  {"k_norm.", {128}}};

  for (int32_t i = 0; i < meta.size(); ++i) {
    all_loaded += load_weight_common(
        state_dict.get_dict_with_prefix("self_attn." + meta[i].first),
        meta[i].second,
        i);
  }
  all_loaded += load_weight_common(
      state_dict.get_dict_with_prefix("input_layernorm."), {hidden_size_}, 6);

  if (all_loaded == weight_num_) {
    all_loaded = 0;
    weights_ = xllm_musa::ReorderAttn(weights_);
  }
}
}  // namespace layer
}  // namespace xllm
