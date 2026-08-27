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

#include "eagle3_decoder_loader.h"

#include <set>

#include "eagle3_loader_constants.h"

namespace xllm {
namespace layer {

using namespace eagle3_decoder_constants;

Eagle3DecoderLoader::Eagle3DecoderLoader(uint64_t weight_count,
                                         const ModelContext& context,
                                         LoadMode mode)
    : BaseLoader(weight_count, context, mode) {
  auto options = context.get_tensor_options();
  device_id_ = options.device().index();

  if (load_to_host()) {
    auto host_options =
        torch::TensorOptions().dtype(options.dtype()).device(torch::kCPU);
    for (int i = 0; i < weight_count; ++i) {
      working_tensors()[i] = torch::zeros({1}, host_options);
    }
  } else {
    for (int i = 0; i < weight_count; ++i) {
      working_tensors()[i] = torch::zeros({1}).to(options);
    }
  }
}

void Eagle3DecoderLoader::load_state_dict(const StateDict& state_dict) {
  const bool to_host = load_to_host();
  auto& w = working_tensors();
  if (quantize_type_ == "w8a8") {
    for (const auto& [index, name] : WEIGHT_MAPPING_W8A8) {
      if (WEIGHT_SHARD_W8A8.find(index) != WEIGHT_SHARD_W8A8.end()) {
        set_weight(state_dict,
                   name,
                   index,
                   WEIGHT_SHARD_W8A8[index],
                   dp_local_tp_rank_,
                   dp_local_tp_size_,
                   to_host);
      } else {
        set_weight(state_dict, name, index, to_host);
      }
    }
    w[IN_NORM_BIAS] =
        torch::zeros(w[IN_NORM_WEIGHT].sizes(), w[IN_NORM_WEIGHT].options());
    w[IN_HIDDEN_NORM_BIAS] = torch::zeros(w[IN_HIDDEN_NORM_WEIGHT].sizes(),
                                          w[IN_HIDDEN_NORM_WEIGHT].options());
    w[IN_SELFOUT_NORM_BIAS] = torch::zeros(w[IN_SELFOUT_NORM_WEIGHT].sizes(),
                                           w[IN_SELFOUT_NORM_WEIGHT].options());
    return;
  }

  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict,
                 name,
                 index,
                 WEIGHT_SHARD[index],
                 dp_local_tp_rank_,
                 dp_local_tp_size_,
                 to_host);
    } else {
      set_weight(state_dict, name, index, to_host);
    }
  }
}

void Eagle3DecoderLoader::merge_host_at_weights() {
  auto& w = working_tensors();

  if (quantize_type_ == "w8a8") {
    w[IN_ATTENTION_OUT_DEQSCALE] =
        w[IN_ATTENTION_OUT_DEQSCALE].to(torch::kFloat32);
    w[IN_Q_DEQSCALE] =
        torch::cat({w[IN_Q_DEQSCALE], w[IN_K_DEQSCALE], w[IN_V_DEQSCALE]}, 0)
            .to(torch::kFloat32);
    w[IN_K_DEQSCALE] = zero_like_working(IN_K_DEQSCALE);
    w[IN_V_DEQSCALE] = zero_like_working(IN_V_DEQSCALE);
    w[IN_K_OFFSET] = zero_like_working(IN_K_OFFSET);
    w[IN_V_OFFSET] = zero_like_working(IN_V_OFFSET);
    w[IN_K_SCALE] = zero_like_working(IN_K_SCALE);
    w[IN_V_SCALE] = zero_like_working(IN_V_SCALE);
    w[IN_MLP_W2_BIAS] = torch::cat({w[IN_MLP_W2_BIAS], w[IN_MLP_W1_BIAS]}, 0);
    w[IN_MLP_W1_BIAS] = zero_like_working(IN_MLP_W1_BIAS);
    w[IN_MLP_W2_DEQSCALE] =
        torch::cat({w[IN_MLP_W2_DEQSCALE], w[IN_MLP_W1_DEQSCALE]}, 0)
            .to(torch::kFloat32);
    w[IN_MLP_W1_DEQSCALE] = zero_like_working(IN_MLP_W1_DEQSCALE);
    w[IN_MLP_W1_OFFSET] = zero_like_working(IN_MLP_W1_OFFSET);
    w[IN_MLP_W1_SCALE] = zero_like_working(IN_MLP_W1_SCALE);
    w[IN_Q_OFFSET] = w[IN_Q_OFFSET].to(torch::kInt8);
    w[IN_ATTENTION_OUT_OFFSET] = w[IN_ATTENTION_OUT_OFFSET].to(torch::kInt8);
    w[IN_MLP_W2_OFFSET] = w[IN_MLP_W2_OFFSET].to(torch::kInt8);
    if (device_id_ != 0) {
      auto original = w[IN_ATTENTION_OUT_BIAS];
      w[IN_ATTENTION_OUT_BIAS] = torch::zeros(original.sizes(),
                                              torch::TensorOptions()
                                                  .dtype(original.dtype())
                                                  .device(target_device()));
    }
  }

  w[IN_Q_WEIGHT] =
      torch::cat({w[IN_Q_WEIGHT], w[IN_K_WEIGHT], w[IN_V_WEIGHT]}, 0)
          .contiguous();
  w[IN_K_WEIGHT] = zero_like_working(IN_K_WEIGHT);
  w[IN_V_WEIGHT] = zero_like_working(IN_V_WEIGHT);

  if (w[IN_Q_BIAS].sizes() != std::vector<int64_t>({1})) {
    w[IN_Q_BIAS] =
        torch::cat({w[IN_Q_BIAS], w[IN_K_BIAS], w[IN_V_BIAS]}, 0).contiguous();
    w[IN_K_BIAS] = zero_like_working(IN_K_BIAS);
    w[IN_V_BIAS] = zero_like_working(IN_V_BIAS);
  }

  TransposeType transpose_type = check_transpose(w[IN_MLP_W2_WEIGHT]);
  if (transpose_type == TransposeType::TRANSPOSE) {
    w[IN_MLP_W2_WEIGHT] =
        torch::cat({w[IN_MLP_W2_WEIGHT], w[IN_MLP_W1_WEIGHT]}, 0).contiguous();
  } else {
    w[IN_MLP_W2_WEIGHT] =
        torch::cat({w[IN_MLP_W2_WEIGHT], w[IN_MLP_W1_WEIGHT]}, 0)
            .transpose(0, 1)
            .contiguous();
  }

  w[IN_MLP_W1_WEIGHT] = zero_like_working(IN_MLP_W1_WEIGHT);
}

TransposeType Eagle3DecoderLoader::check_transpose(at::Tensor& tensor) {
  bool is_k_divisible = tensor.size(1) % 256 == 0;
  bool is_n_divisible = tensor.size(0) % 256 == 0;

  if (!is_k_divisible && is_n_divisible) {
    return TransposeType::NOT_TRANSPOSE;
  }

  return TransposeType::TRANSPOSE;
}

void Eagle3DecoderLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(working_tensors()[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

}  // namespace layer
}  // namespace xllm
