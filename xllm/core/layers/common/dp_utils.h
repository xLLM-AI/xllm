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

#include <cstdint>
#include <utility>
#include <vector>

#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"

namespace xllm {
namespace layer {

// Used to record information before padding
struct PaddingInfo {
  int64_t original_tokens = 0;  // Number of tokens before padding
  int64_t padded_tokens = 0;    // Number of tokens after padding
  bool active = false;          // Whether padding was performed
};

int64_t get_reduce_scatter_tokens(int64_t num_tokens,
                                  const ParallelArgs& parallel_args);

std::pair<torch::Tensor, PaddingInfo> pad_tokens(torch::Tensor x,
                                                 int64_t target_tokens);

// Prepare attention output for TP reduce scatter by applying the rank0
// residual add, padding to target_tokens, then scattering across TP ranks.
std::pair<torch::Tensor, PaddingInfo> reduce_scatter_attn_input(
    torch::Tensor x,
    const torch::Tensor& residual,
    int64_t target_tokens,
    const ParallelArgs& parallel_args);

int64_t get_dp_gather_tokens(const std::vector<int32_t>& dp_tokens,
                             const ParallelArgs& args);

// V2 TP-sharded/global gather path: dp_tokens are model-level token counts and
// data is gathered across the global process group after TP padding.
torch::Tensor gather_global_tokens(const torch::Tensor& input,
                                   const std::vector<int32_t>& dp_tokens,
                                   const ParallelArgs& args);

// Unpadding logic after All Gather
// Simply slice out the original length
torch::Tensor unpad_tokens(torch::Tensor x, const PaddingInfo& pad_info);

// check if decoder should gather dp tokens before moe
bool need_dp_moe_gather(const ParallelArgs& args, bool enable_moe_all2all);

// gather tokens from all dp ranks before moe
torch::Tensor gather_dp_tokens(const torch::Tensor& input,
                               const ModelInputParams& params,
                               const ParallelArgs& args);

struct SelectedMoeInputs {
  torch::Tensor hidden_states;
  torch::Tensor topk_weights;
  torch::Tensor topk_ids;
  bool need_slice = false;
};

// Selected-experts DP-local path: dp_token_nums are explicit first-dimension
// row counts for hidden/topk tensors, and are used without normalization.
bool need_selected_moe_dp_gather(const ParallelArgs& args);

SelectedMoeInputs gather_selected_moe_inputs(
    const torch::Tensor& hidden_states,
    const torch::Tensor& topk_weights,
    const torch::Tensor& topk_ids,
    const std::vector<int32_t>& dp_token_nums,
    const ParallelArgs& args);

torch::Tensor slice_selected_moe_output(
    torch::Tensor output,
    const std::vector<int32_t>& dp_token_nums,
    const ParallelArgs& args);

// given a tensor containing data from all DP ranks,
// returns a slice containing only the tokens for the current DP rank
torch::Tensor get_dp_local_slice(const torch::Tensor& input,
                                 const ModelInputParams& params,
                                 const ParallelArgs& args);

// check if all dp ranks are decode
bool all_dp_ranks_are_decode(const ModelInputParams& params);

}  // namespace layer
}  // namespace xllm
