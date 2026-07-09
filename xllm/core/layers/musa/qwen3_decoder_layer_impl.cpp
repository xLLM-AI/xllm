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

#include "layers/musa/qwen3_decoder_layer_impl.h"

#include "layers/common/rotary_embedding.h"
#include "layers/musa/attention.h"
#include "layers/musa/mlp.h"

namespace xllm::layer {

Qwen3DecoderLayerImpl::Qwen3DecoderLayerImpl(const ModelContext& context) {
  auto const& model_args = context.get_model_args();
  auto const& quant_args = context.get_quant_args();
  auto const& parallel_args = context.get_parallel_args();
  auto const& options = context.get_tensor_options();

  layers_.push_back(register_module(
      "musa_attn", Attention(model_args, quant_args, parallel_args, options)));
  layers_.push_back(register_module("musa_mlp",
                                    MusaMLP(model_args.hidden_size(),
                                            model_args.intermediate_size(),
                                            true,
                                            false,
                                            model_args.hidden_act(),
                                            quant_args,
                                            parallel_args,
                                            options,
                                            model_args.rms_norm_eps())));
}

void Qwen3DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  for (auto&& mod : layers_) {
    mod->load_state_dict(state_dict);
  }
}

torch::Tensor Qwen3DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  // torch::Tensor k_cache = kv_cache.get_k_cache();
  // k_cache = k_cache.view({-1, k_cache.size(1) * 8,  k_cache.size(2)});
  // torch::Tensor v_cache = kv_cache.get_v_cache();
  // v_cache = v_cache.view({-1, v_cache.size(1) * 8,  v_cache.size(2)});

  ForwardParams f{positions, attn_metadata, kv_cache, input_params};

  for (auto&& mod : layers_) {
    x = mod->forward(x, f);
  }

  return x;
}
}  // namespace xllm::layer
