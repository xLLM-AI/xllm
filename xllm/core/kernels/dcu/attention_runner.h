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
#include <functional>
#include <memory>
#include <optional>
#include <tuple>

namespace xllm::layer {
struct AttentionMetadata;
}

namespace xllm::kernel::dcu {

struct AttentionReplayParams {
  uint32_t actual_num_tokens = 0;
  std::shared_ptr<layer::AttentionMetadata> attn_metadata;
};

class AttentionRunner final {
 public:
  using RunFn =
      std::function<std::tuple<torch::Tensor, std::optional<torch::Tensor>>(
          const AttentionReplayParams&)>;

  AttentionRunner() = default;

  AttentionRunner(const AttentionRunner&) = delete;
  AttentionRunner& operator=(const AttentionRunner&) = delete;

  AttentionRunner(AttentionRunner&&) noexcept = default;
  AttentionRunner& operator=(AttentionRunner&&) noexcept = default;

  void run_capture(RunFn run_fn);

  void run_replay(const AttentionReplayParams& params);

 private:
  RunFn run_fn_;
};

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
prefill_with_optional_piecewise_capture(AttentionRunner::RunFn run_fn,
                                        const torch::Tensor& output);

}  // namespace xllm::kernel::dcu
