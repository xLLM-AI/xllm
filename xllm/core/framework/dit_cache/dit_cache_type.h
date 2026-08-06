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

#include <string>
#include <unordered_map>

namespace xllm {

using TensorMap = std::unordered_map<std::string, torch::Tensor>;

struct CacheStepIn {
  int64_t step_id = 0;
  TensorMap tensors;

  CacheStepIn(int64_t step_id) : step_id(step_id) {}
  CacheStepIn(int64_t step_id, const TensorMap& tensors)
      : step_id(step_id), tensors(tensors) {}
};

struct CacheStepOut {
  TensorMap tensors;
  CacheStepOut(const TensorMap& tensors) : tensors(tensors) {}
};

struct CacheBlockIn {
  int64_t block_id = 0;
  TensorMap tensors;

  CacheBlockIn(int64_t block_id) : block_id(block_id) {}
  CacheBlockIn(int64_t block_id, const TensorMap& tensors)
      : block_id(block_id), tensors(tensors) {}
};

struct CacheBlockOut {
  TensorMap tensors;
  CacheBlockOut(const TensorMap& tensors) : tensors(tensors) {}
};

// Per-generation params, refreshed on every forward() before the denoise loop.
struct CacheContext {
  int64_t infer_steps = 0;
  int64_t num_blocks = 0;
};

}  // namespace xllm
