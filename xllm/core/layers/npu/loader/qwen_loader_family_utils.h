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

#include <torch/torch.h>

#include <initializer_list>
#include <vector>

namespace xllm {
namespace layer {
namespace qwen_loader_family_utils {

template <typename ZeroLikeFn>
inline void reset_indices(std::vector<at::Tensor>& tensors,
                          std::initializer_list<int> indices,
                          ZeroLikeFn zero_like) {
  for (int idx : indices) {
    tensors[idx] = zero_like(tensors[idx]);
  }
}

template <typename ZeroLikeFn>
inline void merge_packed_qkv_for_tp(std::vector<at::Tensor>& tensors,
                                    int qkv_weight_idx,
                                    int qkv_bias_idx,
                                    int q_weight_idx,
                                    int k_weight_idx,
                                    int v_weight_idx,
                                    int q_bias_idx,
                                    int k_bias_idx,
                                    int v_bias_idx,
                                    int world_size,
                                    ZeroLikeFn zero_like) {
  if (world_size <= 1) {
    return;
  }

  tensors[qkv_weight_idx] = torch::cat(
      {tensors[q_weight_idx], tensors[k_weight_idx], tensors[v_weight_idx]}, 0);
  reset_indices(tensors, {q_weight_idx, k_weight_idx, v_weight_idx}, zero_like);

  tensors[qkv_bias_idx] = torch::cat(
      {tensors[q_bias_idx], tensors[k_bias_idx], tensors[v_bias_idx]}, 0);
  reset_indices(tensors, {q_bias_idx, k_bias_idx, v_bias_idx}, zero_like);
}

template <typename ZeroLikeFn>
inline void merge_gate_up(std::vector<at::Tensor>& tensors,
                          int gate_weight_idx,
                          int up_weight_idx,
                          int gate_bias_idx,
                          int up_bias_idx,
                          ZeroLikeFn zero_like) {
  tensors[gate_weight_idx] =
      torch::cat({tensors[gate_weight_idx], tensors[up_weight_idx]}, 0);
  tensors[gate_bias_idx] =
      torch::cat({tensors[gate_bias_idx], tensors[up_bias_idx]}, 0);
  tensors[up_bias_idx] = zero_like(tensors[up_bias_idx]);
}

template <typename ZeroLikeFn>
inline void merge_qkv_bias(
    std::vector<at::Tensor>& tensors,
    int q_bias_idx,
    int k_bias_idx,
    int v_bias_idx,
    ZeroLikeFn zero_like,
    c10::optional<torch::ScalarType> dtype = c10::nullopt) {
  auto merged = torch::cat(
      {tensors[q_bias_idx], tensors[k_bias_idx], tensors[v_bias_idx]}, 0);
  tensors[q_bias_idx] = dtype.has_value() ? merged.to(*dtype) : merged;
  reset_indices(tensors, {k_bias_idx, v_bias_idx}, zero_like);
}

template <typename ZeroLikeFn>
inline void merge_qkv_weight_transposed(std::vector<at::Tensor>& tensors,
                                        int q_weight_idx,
                                        int k_weight_idx,
                                        int v_weight_idx,
                                        ZeroLikeFn zero_like) {
  tensors[q_weight_idx] =
      torch::cat(
          {tensors[q_weight_idx], tensors[k_weight_idx], tensors[v_weight_idx]},
          0)
          .transpose(0, 1)
          .contiguous();
  reset_indices(tensors, {k_weight_idx, v_weight_idx}, zero_like);
}

template <typename ZeroLikeFn>
inline void merge_mlp_weight_transposed(std::vector<at::Tensor>& tensors,
                                        int gate_weight_idx,
                                        int up_weight_idx,
                                        ZeroLikeFn zero_like) {
  tensors[gate_weight_idx] =
      torch::cat({tensors[gate_weight_idx], tensors[up_weight_idx]}, 0)
          .transpose(0, 1)
          .contiguous();
  tensors[up_weight_idx] = zero_like(tensors[up_weight_idx]);
}

template <typename ZeroLikeFn>
inline void convert_qwen_decoder_w8a8_shared(std::vector<at::Tensor>& tensors,
                                             int attn_out_deqscale_idx,
                                             int q_deqscale_idx,
                                             int k_deqscale_idx,
                                             int v_deqscale_idx,
                                             int mlp_gate_bias_idx,
                                             int mlp_up_bias_idx,
                                             int mlp_gate_deqscale_idx,
                                             int mlp_up_deqscale_idx,
                                             int q_offset_idx,
                                             int attn_out_offset_idx,
                                             int mlp_gate_offset_idx,
                                             ZeroLikeFn zero_like) {
  tensors[attn_out_deqscale_idx] =
      tensors[attn_out_deqscale_idx].to(torch::kFloat32);
  tensors[q_deqscale_idx] = torch::cat({tensors[q_deqscale_idx],
                                        tensors[k_deqscale_idx],
                                        tensors[v_deqscale_idx]},
                                       0)
                                .to(torch::kFloat32);
  reset_indices(tensors, {k_deqscale_idx, v_deqscale_idx}, zero_like);

  tensors[mlp_gate_bias_idx] =
      torch::cat({tensors[mlp_gate_bias_idx], tensors[mlp_up_bias_idx]}, 0);
  tensors[mlp_up_bias_idx] = zero_like(tensors[mlp_up_bias_idx]);

  tensors[mlp_gate_deqscale_idx] =
      torch::cat({tensors[mlp_gate_deqscale_idx], tensors[mlp_up_deqscale_idx]},
                 0)
          .to(torch::kFloat32);
  tensors[mlp_up_deqscale_idx] = zero_like(tensors[mlp_up_deqscale_idx]);

  tensors[q_offset_idx] = tensors[q_offset_idx].to(torch::kInt8);
  tensors[attn_out_offset_idx] = tensors[attn_out_offset_idx].to(torch::kInt8);
  tensors[mlp_gate_offset_idx] = tensors[mlp_gate_offset_idx].to(torch::kInt8);
}

}  // namespace qwen_loader_family_utils
}  // namespace layer
}  // namespace xllm
