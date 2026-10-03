/* Copyright 2026 The xLLM Authors.

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

#pragma once

#include <torch/torch.h>

#include <cstdint>

namespace xllm::kernel::npu {

inline constexpr int64_t kMegaGdnChunkSize = 128;

struct MegaGdnMasks {
  torch::Tensor mask_lower;
  torch::Tensor mask_full;
  torch::Tensor minus_identity_fp16;
  torch::Tensor minus_identity_bf16;
};

MegaGdnMasks get_or_create_mega_gdn_masks(const torch::Device& device);

}  // namespace xllm::kernel::npu
