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

#include <ATen/DynamicLibrary.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <torch/all.h>

#include <optional>

#include "ATen/Tensor.h"
#include "ATen/cuda/CUDAEvent.h"
#include "c10/core/Device.h"
#include "c10/core/DeviceGuard.h"
#include "c10/core/GradMode.h"
#include "c10/core/InferenceMode.h"
#include "c10/core/MemoryFormat.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "c10/cuda/CUDAFunctions.h"
#include "c10/cuda/CUDAGuard.h"
#include "c10/cuda/CUDAStream.h"
#include "ixformer.h"
#include "kernels/kernels.h"

// #include "utils.h"
using namespace ixformer;

namespace xllm::kernel::ilu {

void apply_rope_pos_ids_cos_sin_cache(torch::Tensor& query,
                                      torch::Tensor& key,
                                      torch::Tensor& cos_sin_cache,
                                      torch::Tensor& positions,
                                      bool interleave);

// act_mode only support silu, gelu, gelu_tanh
void act_and_mul(torch::Tensor out,
                 torch::Tensor input,
                 const std::string& act_mode);

void reshape_paged_cache(
    torch::Tensor& key,                   //  (num_tokens, num_heads, head_size)
    std::optional<torch::Tensor>& value,  // (num_tokens, num_heads, head_size)
    torch::Tensor& key_cache,  // (num_blocks, num_heads, block_size, head_size)
    std::optional<torch::Tensor>&
        value_cache,  // (num_blocks, num_heads, block_size, head_size)
    torch::Tensor& slot_mapping);  //(num_tokens)

void batch_prefill(torch::Tensor& query,
                   const torch::Tensor& key,
                   const std::optional<torch::Tensor>& value,
                   torch::Tensor& output,
                   std::optional<torch::Tensor>& output_lse,
                   const std::optional<torch::Tensor>& q_cu_seq_lens,
                   const std::optional<torch::Tensor>& kv_cu_seq_lens,
                   const std::optional<torch::Tensor>& alibi_slope,
                   const std::optional<torch::Tensor>& attn_bias,
                   const std::optional<torch::Tensor>& q_quant_scale,
                   const std::optional<torch::Tensor>& k_quant_scale,
                   const std::optional<torch::Tensor>& v_quant_scale,
                   const torch::Tensor& block_tables,
                   int64_t max_query_len,
                   int64_t max_seq_len,
                   float scale,
                   bool is_causal,
                   int64_t window_size_left,
                   int64_t window_size_right,
                   const std::string& compute_dtype,
                   bool return_lse);

void batch_decode(torch::Tensor& query,
                  const torch::Tensor& k_cache,
                  torch::Tensor& output,
                  const torch::Tensor& block_table,
                  const torch::Tensor& seq_lens,
                  const std::optional<torch::Tensor>& v_cache,
                  std::optional<torch::Tensor>& output_lse,
                  const std::optional<torch::Tensor>& q_quant_scale,
                  const std::optional<torch::Tensor>& k_cache_quant_scale,
                  const std::optional<torch::Tensor>& v_cache_quant_scale,
                  const std::optional<torch::Tensor>& out_quant_scale,
                  const std::optional<torch::Tensor>& alibi_slope,
                  const std::optional<torch::Tensor>& mask,
                  const std::string& compute_dtype,
                  int64_t max_seq_len,
                  int64_t window_size_left,
                  int64_t window_size_right,
                  float scale,
                  bool return_lse,
                  bool is_causal,
                  int64_t kv_cache_quant_bit_size);

void residual_layer_norm(torch::Tensor& input,
                         torch::Tensor& output,
                         std::optional<torch::Tensor>& residual,
                         torch::Tensor& weight,
                         std::optional<torch::Tensor>& bias,
                         std::optional<torch::Tensor>& residual_out,
                         double eps);

void rms_norm(torch::Tensor& output,
              torch::Tensor& input,
              torch::Tensor& weight,
              double eps);

torch::Tensor matmul(torch::Tensor a,
                     torch::Tensor b,
                     std::optional<torch::Tensor> bias);

std::tuple<torch::Tensor, torch::Tensor> moe_active_topk(
    const torch::Tensor& input,
    int64_t topk,
    int64_t num_expert_group,
    int64_t topk_group,
    bool normalize,
    const std::optional<torch::Tensor>& mask,
    const std::string& normed_by,
    const std::string& scoring_func,
    double route_scale,
    const std::optional<torch::Tensor>& e_score_correction_bias);

std::vector<torch::Tensor> moe_gen_idx(torch::Tensor& expert_id,
                                       int64_t expert_num);

torch::Tensor moe_expand_input(const torch::Tensor& input,
                               const torch::Tensor& gather_index,
                               const torch::Tensor& combine_idx,
                               int64_t topk);

torch::Tensor group_gemm(torch::Tensor& input,
                         torch::Tensor& weight,
                         torch::Tensor& tokens_per_experts,
                         const std::optional<torch::Tensor>& dst_to_src,
                         torch::Tensor& output);

torch::Tensor moe_combine_result(torch::Tensor& input, torch::Tensor& weight);
}  // namespace xllm::kernel::ilu
