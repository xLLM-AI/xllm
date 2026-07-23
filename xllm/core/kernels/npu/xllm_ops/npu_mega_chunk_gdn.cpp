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

#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
#include "triton_npu/torch_api/triton_ops_api.h"

namespace xllm::kernel::npu {

namespace {
constexpr int64_t kMegaChunkSize = 128;

struct MaskCache {
  torch::Tensor mask_lower;
  torch::Tensor mask_full;
  torch::Tensor minus_identity_fp16;
  torch::Tensor minus_identity_bf16;
};

std::unordered_map<int32_t, MaskCache> g_mask_cache;
std::mutex g_mask_cache_mutex;

MaskCache get_or_create_masks(const torch::Device& device) {
  const int32_t device_index = static_cast<int32_t>(device.index());
  std::lock_guard<std::mutex> lock(g_mask_cache_mutex);
  auto it = g_mask_cache.find(device_index);
  if (it != g_mask_cache.end()) {
    return it->second;
  }
  MaskCache cache;
  cache.mask_lower = torch::tril(
      torch::ones({kMegaChunkSize, kMegaChunkSize},
                  torch::TensorOptions(device).dtype(torch::kFloat32)),
      /*diagonal=*/-1);
  cache.mask_full = torch::tril(
      torch::ones({kMegaChunkSize, kMegaChunkSize},
                  torch::TensorOptions(device).dtype(torch::kFloat32)),
      /*diagonal=*/0);
  cache.minus_identity_fp16 =
      torch::zeros({kMegaChunkSize, kMegaChunkSize},
                   torch::TensorOptions(device).dtype(torch::kFloat16));
  cache.minus_identity_fp16.diagonal().fill_(-1);
  cache.minus_identity_bf16 =
      torch::zeros({kMegaChunkSize, kMegaChunkSize},
                   torch::TensorOptions(device).dtype(torch::kBFloat16));
  cache.minus_identity_bf16.diagonal().fill_(-1);
  g_mask_cache[device_index] = cache;
  return cache;
}
}  // namespace

std::pair<torch::Tensor, torch::Tensor> npu_mega_chunk_gdn(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& g,
    torch::Tensor& beta,
    const std::optional<float>& scale,
    const std::optional<torch::Tensor>& initial_state,
    bool output_final_state,
    const std::optional<torch::Tensor>& cu_seqlens,
    c10::ArrayRef<int32_t> q_seq_lens,
    bool use_qk_l2norm_in_kernel) {
  const torch::ScalarType input_dtype = q.scalar_type();

  torch::Tensor q_normalized = q;
  torch::Tensor k_normalized = k;
  if (use_qk_l2norm_in_kernel) {
    q_normalized = npu_l2norm_last_dim(q);
    k_normalized = npu_l2norm_last_dim(k);
  }

  const bool use_bf16_compute =
      q_normalized.scalar_type() == torch::kBFloat16 &&
      k_normalized.scalar_type() == torch::kBFloat16 &&
      v.scalar_type() == torch::kBFloat16;
  const auto compute_dtype =
      use_bf16_compute ? torch::kBFloat16 : torch::kFloat16;
  auto q_compute = q_normalized.to(compute_dtype);
  auto k_compute = k_normalized.to(compute_dtype);
  auto v_compute = v.to(compute_dtype);
  auto g_fp32 = g.to(torch::kFloat32);
  auto beta_compute = beta.to(compute_dtype);

  torch::Tensor cu_seqlens_int32;
  int64_t num_sequences = 0;
  int64_t num_chunks = 0;
  if (cu_seqlens.has_value() && cu_seqlens->defined()) {
    cu_seqlens_int32 = cu_seqlens->to(torch::kInt32);
    if (!q_seq_lens.empty()) {
      num_sequences = static_cast<int64_t>(q_seq_lens.size());
      CHECK_EQ(cu_seqlens_int32.numel(), num_sequences + 1)
          << "cu_seqlens and q_seq_lens must describe the same sequences.";
      for (const int32_t seq_len : q_seq_lens) {
        CHECK_GE(seq_len, 0) << "q_seq_lens must be non-negative.";
        num_chunks += (static_cast<int64_t>(seq_len) + kMegaChunkSize - 1) /
                      kMegaChunkSize;
      }
    } else {
      num_sequences = cu_seqlens_int32.numel() - 1;
      auto cu_cpu = cu_seqlens_int32.to(torch::kCPU);
      auto cu_data = cu_cpu.accessor<int32_t, 1>();
      for (int64_t i = 0; i < num_sequences; ++i) {
        const int64_t seq_len = cu_data[i + 1] - cu_data[i];
        num_chunks += (seq_len + kMegaChunkSize - 1) / kMegaChunkSize;
      }
    }
  } else {
    const int64_t total_tokens = q.size(1);
    cu_seqlens_int32 =
        torch::tensor({0, static_cast<int32_t>(total_tokens)},
                      torch::TensorOptions(q.device()).dtype(torch::kInt32));
    num_sequences = 1;
    num_chunks = (total_tokens + kMegaChunkSize - 1) / kMegaChunkSize;
  }

  const int64_t num_value_heads = v.size(2);
  const int64_t num_matrices = num_chunks * num_value_heads;

  auto masks = get_or_create_masks(q.device());
  const auto& minus_identity =
      use_bf16_compute ? masks.minus_identity_bf16 : masks.minus_identity_fp16;

  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t K = q.size(3);
  const int64_t H = v.size(2);
  const int64_t V = v.size(3);

  // Calculate default scale if not provided
  float scale_value = scale.has_value()
                          ? scale.value()
                          : std::pow(static_cast<float>(K), -0.5f);

  auto opts_compute = torch::TensorOptions(q.device()).dtype(compute_dtype);
  auto opts_fp32 = torch::TensorOptions(q.device()).dtype(torch::kFloat32);

  auto out = torch::empty({B, T, H, V}, opts_compute);
  auto g_sum = torch::empty({B, T, H}, opts_fp32);
  auto g_t = torch::empty({H, T}, opts_fp32);
  auto beta_t = torch::empty({H, T}, opts_compute);
  auto a = torch::zeros({B, T, H, kMegaChunkSize}, opts_compute);
  auto a_inv_f32 = torch::zeros({B, T, H, kMegaChunkSize}, opts_fp32);
  auto a_inv = torch::zeros({B, T, H, kMegaChunkSize}, opts_compute);
  auto w = torch::empty({B, T, H, V}, opts_compute);
  auto u = torch::empty({B, T, H, V}, opts_compute);
  auto h = torch::zeros({num_matrices, K, V}, opts_compute);
  auto v_new = torch::empty({B, T, H, V}, opts_compute);

  torch::Tensor initial_state_arg;
  bool has_initial_state = false;
  if (initial_state.has_value() && initial_state->defined()) {
    initial_state_arg = initial_state->to(compute_dtype);
    has_initial_state = true;
  } else {
    initial_state_arg = torch::zeros({num_sequences, H, K, V}, opts_compute);
  }

  auto final_state = torch::zeros({num_sequences * H, K, V}, opts_compute);

  EXEC_NPU_CMD(aclnnMegaChunkGdn,
               q_compute,
               k_compute,
               v_compute,
               g_fp32,
               beta_compute,
               masks.mask_lower,
               masks.mask_full,
               minus_identity,
               cu_seqlens_int32,
               initial_state_arg,
               num_matrices,
               has_initial_state,
               out,
               g_sum,
               g_t,
               beta_t,
               a,
               a_inv_f32,
               a_inv,
               w,
               u,
               h,
               v_new,
               final_state);

  auto output = (out * scale_value).to(input_dtype);

  torch::Tensor final_state_out;
  if (output_final_state) {
    final_state_out =
        final_state.view({num_sequences, H, K, V}).to(torch::kFloat32);
  }

  return {output, final_state_out};
}

}  // namespace xllm::kernel::npu
