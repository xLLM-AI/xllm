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

#include "beam_searcher.h"

#include "common/macros.h"
#if defined(USE_MLU)
#include "kernels/mlu/mlu_ops_api.h"
#endif

namespace xllm {
BeamSearchOutput BeamSearcher::forward(const torch::Tensor& logprobs,
                                       const torch::Tensor& top_tokens,
                                       const torch::Tensor& top_logprobs,
                                       int64_t beam_width) const {
#if defined(USE_NPU)
  BeamSearchOutput output;

  int64_t num_seq = logprobs.numel();
  output.out_tokens =
      torch::empty({num_seq, 1}, logprobs.options().dtype(torch::kInt32));
  output.out_logprobs =
      torch::empty({num_seq, 1}, logprobs.options().dtype(torch::kFloat32));
  output.src_seq_idxes =
      torch::empty({num_seq, 1}, logprobs.options().dtype(torch::kInt32));
  xllm::kernel::npu::beam_search(logprobs.reshape({-1, 1}),
                                 top_tokens.to(torch::kInt32),
                                 top_logprobs,
                                 output.src_seq_idxes,
                                 output.out_logprobs,
                                 output.out_tokens);
  output.src_seq_idxes = output.src_seq_idxes.reshape({-1});
  output.out_logprobs = output.out_logprobs.reshape({-1});
  output.out_tokens = output.out_tokens.reshape({-1});
  return output;
#elif defined(USE_MLU)
  // Guard against callers that do not supply beam_width (e.g. rec engine),
  // which would otherwise divide by zero in the device kernel.
  if (beam_width <= 0) {
    return BeamSearchOutput();
  }
  return xllm::kernel::mlu::beam_search(
      logprobs, top_tokens, top_logprobs, beam_width);
#else
  NOT_IMPLEMENTED();
#endif
}
}  // namespace xllm
