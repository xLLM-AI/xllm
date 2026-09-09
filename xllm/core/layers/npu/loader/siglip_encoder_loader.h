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

class SiglipEncoderUpLoader : public BaseLoader {
 public:
  explicit SiglipEncoderUpLoader(const ModelContext& context);

  void load_state_dict(const StateDict& state_dict) override;
};

class SiglipEncoderDownLoader : public BaseLoader {
 public:
  explicit SiglipEncoderDownLoader(const ModelContext& context);

  void load_state_dict(const StateDict& state_dict) override;
};

}  // namespace layer
}  // namespace xllm
