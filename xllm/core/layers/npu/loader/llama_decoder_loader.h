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

#include <map>
#include <vector>

#include "base_loader.h"

namespace xllm {
namespace layer {

class LlamaDecoderLoader : public BaseLoader {
 public:
  LlamaDecoderLoader(uint64_t weight_count,
                     const ModelContext& context,
                     LoadMode mode = LoadMode::kEager);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights() const override;

 protected:
  void merge_host_at_weights() override;

  bool enableAddNorm_ = false;
  int rank_id_ = 0;
};

}  // namespace layer
}  // namespace xllm
