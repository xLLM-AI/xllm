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
#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "models/llm/mlu/mtp_topk_state.h"
#include "models/llm/mtp_model_base.h"

namespace xllm::mlu::model {

// Adds the MLU decoder's typed DSA transfer to the shared MTP projection and
// normalization implementation without exposing MLU payloads to that base.
template <typename DecoderLayerType>
class MluMtpDecoderLayerImplBase
    : public ::xllm::MtpDecoderLayerImplBase<DecoderLayerType> {
 public:
  using Base = ::xllm::MtpDecoderLayerImplBase<DecoderLayerType>;

  MluMtpDecoderLayerImplBase(const ModelContext& context, int32_t layer_index)
      : Base(context, layer_index) {}

  torch::Tensor forward_with_mtp_topk(
      torch::Tensor embed,
      std::optional<torch::Tensor>& residual,
      torch::Tensor positions,
      const layer::AttentionMetadata& attn_metadata,
      KVCache& kv_cache,
      ModelInputParams& input_params,
      const std::optional<torch::Tensor>& input_ids,
      const std::optional<layer::DsaTopkState>& topk_input,
      std::optional<layer::DsaTopkState>& topk_output) {
    auto decoder_forward = [&topk_input, &topk_output](
                               DecoderLayerType& decoder,
                               torch::Tensor& hidden_states,
                               std::optional<torch::Tensor>& layer_residual,
                               torch::Tensor& layer_positions,
                               const layer::AttentionMetadata& metadata,
                               KVCache& layer_kv_cache,
                               ModelInputParams& params,
                               const std::optional<torch::Tensor>& ids) {
      return decoder->forward_mtp(hidden_states,
                                  layer_residual,
                                  layer_positions,
                                  metadata,
                                  layer_kv_cache,
                                  params,
                                  ids,
                                  topk_input,
                                  topk_output);
    };
    return Base::forward_with_decoder(std::move(embed),
                                      residual,
                                      std::move(positions),
                                      attn_metadata,
                                      kv_cache,
                                      input_params,
                                      input_ids,
                                      decoder_forward);
  }
};

// Model-level policy used by the shared MTP forward loop. It maps the single
// worker-facing state to per-layer MLU DSA state and collects the next state.
template <typename DecoderLayerType>
class MluMtpLayerForwardAdapter final {
 public:
  MluMtpLayerForwardAdapter(const MtpTopkStatePtr& input_state,
                            size_t num_layers,
                            const torch::Device& /*device*/) {
    topk_outputs_.reserve(num_layers);
    if (input_state == nullptr) {
      return;
    }
    input_state_ =
        std::dynamic_pointer_cast<const MluMtpTopkState>(input_state);
    CHECK(input_state_ != nullptr)
        << "MLU MTP model received an incompatible top-k state.";
    CHECK_EQ(input_state_->layer_states().size(), num_layers)
        << "MLU MTP top-k state count must match decoder layer count.";
  }

  torch::Tensor forward_layer(DecoderLayerType& decoder_layer,
                              size_t layer_id,
                              torch::Tensor& hidden_states,
                              std::optional<torch::Tensor>& residual,
                              torch::Tensor& positions,
                              const layer::AttentionMetadata& attn_metadata,
                              KVCache& kv_cache,
                              ModelInputParams& input_params,
                              const std::optional<torch::Tensor>& input_ids) {
    const std::optional<layer::DsaTopkState>& topk_input =
        input_state_ == nullptr ? empty_topk_state_
                                : input_state_->layer_states()[layer_id];
    std::optional<layer::DsaTopkState> topk_output;
    torch::Tensor output = decoder_layer->forward_with_mtp_topk(hidden_states,
                                                                residual,
                                                                positions,
                                                                attn_metadata,
                                                                kv_cache,
                                                                input_params,
                                                                input_ids,
                                                                topk_input,
                                                                topk_output);
    topk_outputs_.emplace_back(std::move(topk_output));
    return output;
  }

  MtpTopkStatePtr take_output() {
    const bool has_topk_output =
        std::any_of(topk_outputs_.begin(),
                    topk_outputs_.end(),
                    [](const std::optional<layer::DsaTopkState>& state) {
                      return state.has_value();
                    });
    if (!has_topk_output) {
      return nullptr;
    }
    return std::make_shared<MluMtpTopkState>(std::move(topk_outputs_));
  }

 private:
  std::shared_ptr<const MluMtpTopkState> input_state_;
  std::optional<layer::DsaTopkState> empty_topk_state_;
  MluMtpTopkState::LayerStates topk_outputs_;
};

template <typename DecoderLayerType>
using MluMtpModelImplBase =
    ::xllm::MtpModelImplBase<DecoderLayerType,
                             MluMtpLayerForwardAdapter<DecoderLayerType>>;

}  // namespace xllm::mlu::model
