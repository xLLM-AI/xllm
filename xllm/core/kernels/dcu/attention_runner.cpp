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

#include "kernels/dcu/attention_runner.h"

#include <c10/core/InferenceMode.h>
#include <glog/logging.h>

#include <limits>
#include <utility>

#include "kernels/dcu/global_capture_instance.h"

namespace xllm::kernel::dcu {

void AttentionRunner::run_capture(RunFn run_fn) {
  auto& capture = ::xllm::runtime::dcu::GlobalCaptureInstance::get_instance();

  capture.temporarily_end_graph();

  run_fn_ = std::move(run_fn);
  CHECK(run_fn_ != nullptr);

  // Match the CUDA runner behavior: capture records replay state without
  // executing attention.
  capture.temporarily_begin_graph();
}

void AttentionRunner::run_replay(const AttentionReplayParams& params) {
  CHECK(run_fn_ != nullptr);

  c10::InferenceMode guard(true);
  run_fn_(params);
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
prefill_with_optional_piecewise_capture(AttentionRunner::RunFn run_fn,
                                        const torch::Tensor& output) {
  auto& capture = ::xllm::runtime::dcu::GlobalCaptureInstance::get_instance();

  if (capture.is_capturing()) {
    AttentionRunner runner;
    runner.run_capture(std::move(run_fn));

    capture.register_attention_runner(std::move(runner));

    // Capture skips prefill attention and returns the output placeholder so the
    // following graph segment records the same output address.
    return {output, std::nullopt};
  }

  AttentionReplayParams params;
  const int64_t actual_num_tokens = output.size(0);
  CHECK_GE(actual_num_tokens, 0);
  CHECK_LE(actual_num_tokens, std::numeric_limits<uint32_t>::max());
  params.actual_num_tokens = static_cast<uint32_t>(actual_num_tokens);
  return run_fn(params);
}

}  // namespace xllm::kernel::dcu
