/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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
#include <map>
#include <vector>

#include "base_loader.h"

namespace xllm {
namespace layer {

class Kimik25VisionEncoderLoader : public BaseLoader {
 public:
  Kimik25VisionEncoderLoader(uint64_t weight_count,
                             const ModelContext& context);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights() const override;
  void merge_loaded_weights() override;

 private:
  void get_weights_col_packed_qkv();

 protected:
  std::string model_name_;
  at::Tensor cu_seqlen_;
  at::Tensor at_placeholder_;
  std::vector<torch::Tensor> qkv_weight_;
  std::vector<torch::Tensor> qkv_bias_;
  int32_t device_id_;
  int32_t encode_param_rank_;
  int32_t encode_param_worldSize_;
};

}  // namespace layer
}  // namespace xllm
