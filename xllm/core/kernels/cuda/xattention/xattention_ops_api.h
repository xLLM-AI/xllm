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

#include <vector>

namespace xllm::kernel::cuda {

void decoder_reshape_and_cache(torch::Tensor proj_k,
                               torch::Tensor proj_v,
                               torch::Tensor unshared_k_cache,
                               torch::Tensor unshared_v_cache,
                               torch::Tensor step);

void cache_select(const torch::Tensor& beam_index,
                  std::vector<torch::Tensor>& unshared_k_cache,
                  std::vector<torch::Tensor>& unshared_v_cache,
                  const torch::Tensor& block_table,
                  int64_t decode_step,
                  int64_t beam_size,
                  int64_t layer_num);

void lse_combine(torch::Tensor output,
                 torch::Tensor shared_o,
                 torch::Tensor shared_lse,
                 torch::Tensor unshared_o,
                 torch::Tensor unshared_lse);

void prefill_reshape_and_cache(
    torch::Tensor proj_k,  // [shared_len, kv_heads, head_dim]
    torch::Tensor proj_v,  // [shared_len, kv_heads, head_dim]
    torch::Tensor
        shared_k_cache,  // [num_shared_kv_seq_len, kv_heads, head_dim]
    torch::Tensor shared_v_cache);

void beam_search(torch::Tensor acc_logprob,
                 torch::Tensor in_sequence_group,
                 torch::Tensor top_tokens,
                 torch::Tensor top_logprobs,
                 torch::Tensor out_acc_logprob,
                 torch::Tensor out_token_ids,
                 torch::Tensor out_token_index,
                 torch::Tensor out_beam_count_prefix_sums,
                 torch::Tensor out_sequence_group,
                 uint32_t batch_size,
                 uint32_t num_return_sequences,
                 uint32_t current_step);

}  // namespace xllm::kernel::cuda
