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

#include <torch/torch.h>

#include <cstdint>
#include <memory>

namespace xllm {

class MtpTopkState;
using MtpTopkStatePtr = std::shared_ptr<const MtpTopkState>;

// Backend-neutral state carried between MTP draft steps. Runtime code owns
// row-selection orchestration; backend implementations only expose the common
// row axis and interpret index_select_rows() for their native payload.
class MtpTopkState {
 public:
  virtual ~MtpTopkState() = default;

  virtual int64_t num_rows() const = 0;
  virtual torch::Device device() const = 0;
  virtual MtpTopkStatePtr to(const torch::Device& device) const = 0;
  virtual MtpTopkStatePtr index_select_rows(
      const torch::Tensor& index) const = 0;
};

}  // namespace xllm
