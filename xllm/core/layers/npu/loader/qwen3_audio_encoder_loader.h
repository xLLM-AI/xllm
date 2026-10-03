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

#include <cstdint>

#include "core/layers/npu/loader/base_loader.h"

namespace xllm::layer {

class Qwen3AudioEncoderLoader final : public BaseLoader {
 public:
  Qwen3AudioEncoderLoader(uint64_t weight_count, const ModelContext& context);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights() const override;
  void merge_loaded_weights() override;

 private:
  void get_weights_col_packed_qkv();

  int32_t encode_param_rank_ = 0;
  int32_t encode_param_world_size_ = 1;
};

}  // namespace xllm::layer
