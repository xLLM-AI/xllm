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

#include "multi_head_attention.h"

namespace xllm {
namespace layer {

MultiheadAttentionImpl::MultiheadAttentionImpl(const ModelContext& context)
    : n_head_(context.get_model_args().n_heads()),
      head_dim_(context.get_model_args().head_dim()),
      hidden_size_(context.get_model_args().hidden_size()),
      options_(context.get_tensor_options()) {}

torch::Tensor MultiheadAttentionImpl::forward(torch::Tensor q,
                                              torch::Tensor k,
                                              torch::Tensor v,
                                              torch::Tensor key_padding_mask) {
  namespace F = torch::nn::functional;
  // get [w_q, w_k, w_v] from in_proj_weight_
  std::vector<torch::Tensor> weight_chunks = in_proj_weight_.chunk(3, 0);
  // get [b_q, b_k, b_v] from in_proj_bias_
  std::vector<torch::Tensor> bias_chunks = in_proj_bias_.chunk(3, 0);

  q = F::linear(q, weight_chunks[0], bias_chunks[0]);
  k = F::linear(k, weight_chunks[1], bias_chunks[1]);
  v = F::linear(v, weight_chunks[2], bias_chunks[2]);

  // reshape q, k, v for multihead attention
  auto tgt_len = q.size(0);
  q = q.view({q.size(0), q.size(1) * n_head_, head_dim_}).transpose(0, 1);
  k = k.view({k.size(0), k.size(1) * n_head_, head_dim_}).transpose(0, 1);
  v = v.view({v.size(0), v.size(1) * n_head_, head_dim_}).transpose(0, 1);

  // build attention mask
  auto batch_size = key_padding_mask.size(0);
  auto src_len = key_padding_mask.size(1);
  key_padding_mask = torch::zeros_like(key_padding_mask, options_.dtype())
                         .masked_fill_(key_padding_mask,
                                       -std::numeric_limits<float>::infinity());
  key_padding_mask = key_padding_mask.view({batch_size, 1, 1, src_len})
                         .expand({batch_size, n_head_, 1, src_len})
                         .reshape({batch_size * n_head_, 1, src_len});

  // multihead attention
  q = q / std::sqrt(static_cast<float>(head_dim_));
  auto attn_output = torch::matmul(q, k.transpose(-2, -1)) + key_padding_mask;
  attn_output = torch::softmax(attn_output, -1);
  attn_output = torch::matmul(attn_output, v);

  attn_output = attn_output.transpose(0, 1).contiguous().view(
      {tgt_len, batch_size, hidden_size_});
  attn_output = F::linear(attn_output, out_proj_weight_, out_proj_bias_);

  return attn_output;
}

void MultiheadAttentionImpl::load_state_dict(const StateDict& state_dict) {
  const auto in_proj_weight = state_dict.get_tensor("in_proj_weight");
  if (in_proj_weight.defined()) {
    in_proj_weight_ = in_proj_weight.to(options_);
    in_proj_weight_is_loaded_ = true;
  }

  const auto in_proj_bias = state_dict.get_tensor("in_proj_bias");
  if (in_proj_bias.defined()) {
    in_proj_bias_ = in_proj_bias.to(options_);
    in_proj_bias_is_loaded_ = true;
  }

  const auto out_proj_weight = state_dict.get_tensor("out_proj.weight");
  if (out_proj_weight.defined()) {
    out_proj_weight_ = out_proj_weight.to(options_);
    out_proj_weight_is_loaded_ = true;
  }

  const auto out_proj_bias = state_dict.get_tensor("out_proj.bias");
  if (out_proj_bias.defined()) {
    out_proj_bias_ = out_proj_bias.to(options_);
    out_proj_bias_is_loaded_ = true;
  }
}

void MultiheadAttentionImpl::verify_loaded_weights(
    const std::string& prefix) const {
  CHECK(in_proj_weight_is_loaded_)
      << "in_proj_weight is not loaded for " << prefix + "in_proj_weight";
  CHECK(in_proj_bias_is_loaded_)
      << "in_proj_bias is not loaded for " << prefix + "in_proj_bias";
  CHECK(out_proj_weight_is_loaded_)
      << "out_proj.weight is not loaded for " << prefix + "out_proj.weight";
  CHECK(out_proj_bias_is_loaded_)
      << "out_proj.bias is not loaded for " << prefix + "out_proj.bias";
}

}  // namespace layer
}  // namespace xllm
