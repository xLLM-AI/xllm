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

#pragma once

#include <torch/torch.h>

#include <vector>

#include "common/macros.h"

namespace xllm {
namespace xtensor {
struct Options {
  // devices for xtensor manager pool
  PROPERTY(std::vector<torch::Device>, devices);

  // Multi-node configuration
  PROPERTY(int32_t, nnodes) = 1;
  PROPERTY(int32_t, node_rank) = 0;
};
}  // namespace xtensor
}  // namespace xllm
