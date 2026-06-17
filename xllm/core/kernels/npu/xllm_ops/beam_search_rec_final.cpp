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

#include <glog/logging.h>
#include <torch/torch.h>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/utils.h"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

void run_beam_search_rec_final(const torch::Tensor& logprobs,
                               const torch::Tensor& top_tokens,
                               const torch::Tensor& top_logprobs,
                               torch::Tensor& sequence_group,
                               int64_t current_step,
                               int64_t result_width,
                               torch::Tensor& out_token_ids,
                               torch::Tensor& out_token_index,
                               torch::Tensor& out_log_probs,
                               torch::Tensor& out_beam_count_prefix_sums,
                               torch::Tensor& out_sequence) {
  CHECK_GT(result_width, 0)
      << "beam_search_rec final select requires positive result_width";
  CHECK_EQ(out_sequence.dim(), 3)
      << "beam_search_rec final select expects 3D out_sequence";
  CHECK_EQ(out_sequence.size(1), result_width)
      << "beam_search_rec final select output width mismatch";
  check_tensor(logprobs, "logprobs", "beam_search_rec_final");
  check_tensor(top_tokens, "top_tokens", "beam_search_rec_final");
  check_tensor(top_logprobs, "top_logprobs", "beam_search_rec_final");
  check_tensor(sequence_group, "sequence_group", "beam_search_rec_final");
  check_tensor(out_token_ids, "out_token_ids", "beam_search_rec_final");
  check_tensor(out_token_index, "out_token_index", "beam_search_rec_final");
  check_tensor(out_log_probs, "out_log_probs", "beam_search_rec_final");
  check_tensor(out_beam_count_prefix_sums,
               "out_beam_count_prefix_sums",
               "beam_search_rec_final");
  check_tensor(out_sequence, "out_sequence", "beam_search_rec_final");

  // The final widened result is returned to host only, so cache-select prefix
  // metadata is not consumed after this kernel.
  out_beam_count_prefix_sums.zero_();

  EXEC_NPU_CMD(aclnnOnerecFinalBeamSelect,
               logprobs,
               top_tokens,
               top_logprobs,
               sequence_group,
               current_step,
               out_token_ids,
               out_token_index,
               out_log_probs,
               out_sequence);
}

}  // namespace xllm::kernel::npu
