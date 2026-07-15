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

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace xllm::npu {

class VisionEncoderGraphAdapter {
 public:
  virtual ~VisionEncoderGraphAdapter() = default;

  virtual int32_t num_encoder_layers() const = 0;

  virtual const std::vector<int64_t>& deepstack_indexes() const = 0;

  virtual torch::Tensor forward_encoder_layer(
      int32_t layer_idx,
      torch::Tensor& hidden,
      torch::Tensor& cos_pos,
      torch::Tensor& sin_pos,
      torch::Tensor& cu_seqlens,
      std::vector<int>& cu_seqlens_vec) = 0;

  virtual torch::Tensor forward_deepstack_merger(
      int32_t deepstack_slot,
      const torch::Tensor& hidden) = 0;
};

}  // namespace xllm::npu
