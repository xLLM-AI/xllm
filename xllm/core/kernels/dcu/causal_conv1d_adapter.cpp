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

#include "causal_conv1d_adapter.h"

#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>

#include "causal_conv1d/causal_conv1d.h"

// Template kernel forward declarations from the copied .hip files.
template <typename input_t, typename weight_t>
void causal_conv1d_channellast_fwd_cuda(ConvParamsBase& params,
                                        hipStream_t stream);
template <typename input_t, typename weight_t>
void causal_conv1d_varlen_channellast_fwd_cuda(ConvParamsBase& params,
                                               hipStream_t stream);
template <typename input_t, typename weight_t>
void causal_conv1d_update_cuda(ConvParamsBase& params, hipStream_t stream);

namespace xllm {
namespace kernel {
namespace dcu {

namespace {

// Fill common fields of ConvParamsBase for forward passes.
// `varlen=true` selects the packed-varlen stride layout for x/out.
void set_conv_params_fwd(ConvParamsBase& params,
                         int64_t batch,
                         int64_t dim,
                         int64_t seqlen,
                         int64_t width,
                         const torch::Tensor& x,
                         const torch::Tensor& weight,
                         const torch::Tensor& out,
                         void* bias_ptr,
                         bool silu_activation,
                         bool varlen) {
  std::memset(&params, 0, sizeof(params));
  params.batch = static_cast<int>(batch);
  params.dim = static_cast<int>(dim);
  params.seqlen = static_cast<int>(seqlen);
  params.width = static_cast<int>(width);
  params.silu_activation = silu_activation;
  params.pad_slot_id = -1;

  params.x_ptr = x.data_ptr();
  params.weight_ptr = weight.data_ptr();
  params.bias_ptr = bias_ptr;
  params.out_ptr = out.data_ptr();

  params.x_batch_stride = varlen ? 0 : x.stride(0);
  params.x_c_stride = varlen ? x.stride(0) : x.stride(1);
  params.x_l_stride = varlen ? x.stride(1) : x.stride(-1);
  params.weight_c_stride = weight.stride(0);
  params.weight_width_stride = weight.stride(1);
  params.out_batch_stride = varlen ? 0 : out.stride(0);
  params.out_c_stride = varlen ? out.stride(0) : out.stride(1);
  params.out_l_stride = varlen ? out.stride(1) : out.stride(-1);
}

#define DISPATCH_ITYPE(TYPE, NAME, ...)                    \
  do {                                                     \
    if ((TYPE) == at::ScalarType::Half) {                  \
      using input_t = at::Half;                            \
      __VA_ARGS__();                                       \
    } else if ((TYPE) == at::ScalarType::BFloat16) {       \
      using input_t = at::BFloat16;                        \
      __VA_ARGS__();                                       \
    } else if ((TYPE) == at::ScalarType::Float) {          \
      using input_t = float;                               \
      __VA_ARGS__();                                       \
    } else {                                               \
      TORCH_CHECK(false, NAME " unsupported input dtype"); \
    }                                                      \
  } while (0)

#define DISPATCH_WTYPE(TYPE, NAME, ...)                     \
  do {                                                      \
    if ((TYPE) == at::ScalarType::Half) {                   \
      using weight_t = at::Half;                            \
      __VA_ARGS__();                                        \
    } else if ((TYPE) == at::ScalarType::BFloat16) {        \
      using weight_t = at::BFloat16;                        \
      __VA_ARGS__();                                        \
    } else if ((TYPE) == at::ScalarType::Float) {           \
      using weight_t = float;                               \
      __VA_ARGS__();                                        \
    } else {                                                \
      TORCH_CHECK(false, NAME " unsupported weight dtype"); \
    }                                                       \
  } while (0)

}  // namespace

torch::Tensor causal_conv1d_varlen_fwd(
    const torch::Tensor& x_tc,
    const torch::Tensor& weight_ck,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& state_indices,
    torch::Tensor& conv_cache,
    bool activation) {
  TORCH_CHECK(x_tc.dim() == 2, "x_tc must be [T, C]");
  TORCH_CHECK(weight_ck.dim() == 2, "weight_ck must be [C, K]");
  TORCH_CHECK(conv_cache.dim() == 3, "conv_cache must be [N, K-1, C]");
  TORCH_CHECK(cu_seqlens.size() >= 2, "cu_seqlens needs N+1 entries");
  const int64_t N = static_cast<int64_t>(cu_seqlens.size()) - 1;
  TORCH_CHECK(static_cast<int64_t>(state_indices.size()) == N,
              "state_indices size must equal N");
  TORCH_CHECK(x_tc.dtype() == weight_ck.dtype(),
              "x and weight must share dtype");
  TORCH_CHECK(x_tc.dtype() == conv_cache.dtype(),
              "x and conv_cache must share dtype");

  const int64_t T_total = x_tc.size(0);
  const int64_t C = x_tc.size(1);
  const int64_t K = weight_ck.size(1);
  TORCH_CHECK(weight_ck.size(0) == C, "weight_ck[0] must equal C");
  TORCH_CHECK(conv_cache.size(1) == K - 1 && conv_cache.size(2) == C,
              "conv_cache shape must be [N, K-1, C]");
  TORCH_CHECK(K >= 2 && K <= 4, "kernel supports width in [2, 4]");
  TORCH_CHECK(C % 8 == 0, "channel dim must be divisible by 8");

  auto device = x_tc.device();
  c10::DeviceGuard guard(device);

  // Reinterpret x/out as channel-last [C, T] views (stride(0)==1).
  auto x_ct = x_tc.transpose(0, 1);
  auto out_tc = torch::empty_like(x_tc);
  auto out_ct = out_tc.transpose(0, 1);
  // Reinterpret conv_cache to [N, C, K-1] view; stride(1)==1 (channel-first).
  auto conv_state = conv_cache.transpose(1, 2);

  auto int32_opts = torch::TensorOptions().dtype(torch::kInt32).device(device);
  std::vector<int32_t> cu32(cu_seqlens.begin(), cu_seqlens.end());
  auto query_start_loc = torch::tensor(cu32, int32_opts);
  std::vector<int32_t> idx32(state_indices.begin(), state_indices.end());
  auto cache_indices = torch::tensor(idx32, int32_opts);

  // Determine max seqlen for grid sizing.
  int64_t max_seqlen = 0;
  for (int64_t i = 0; i < N; ++i) {
    max_seqlen = std::max(max_seqlen, cu_seqlens[i + 1] - cu_seqlens[i]);
  }

  ConvParamsBase params;
  set_conv_params_fwd(params,
                      N,
                      C,
                      T_total,
                      K,
                      x_ct,
                      weight_ck,
                      out_ct,
                      /*bias_ptr=*/nullptr,
                      activation,
                      /*varlen=*/true);
  params.max_seqlen = static_cast<int>(max_seqlen);
  params.query_start_loc_ptr = query_start_loc.data_ptr<int32_t>();
  params.conv_state_indices_ptr = cache_indices.data_ptr<int32_t>();
  params.conv_state_indices_stride = cache_indices.stride(0);

  params.initial_states_ptr = conv_state.data_ptr();
  params.initial_states_batch_stride = conv_state.stride(0);
  params.initial_states_c_stride = conv_state.stride(1);
  params.initial_states_l_stride = conv_state.stride(2);
  params.conv_state_len = static_cast<int>(K - 1);

  auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  DISPATCH_ITYPE(x_tc.scalar_type(), "causal_conv1d_varlen_fwd", [&] {
    DISPATCH_WTYPE(weight_ck.scalar_type(), "causal_conv1d_varlen_fwd", [&] {
      causal_conv1d_varlen_channellast_fwd_cuda<input_t, weight_t>(params,
                                                                   stream);
    });
  });
  return out_tc;
}

torch::Tensor causal_conv1d_update(const torch::Tensor& x_bc,
                                   const torch::Tensor& weight_ck,
                                   const torch::Tensor& state_indices,
                                   torch::Tensor& conv_cache,
                                   bool activation) {
  TORCH_CHECK(x_bc.dim() == 2, "x_bc must be [B, C]");
  TORCH_CHECK(weight_ck.dim() == 2, "weight_ck must be [C, K]");
  TORCH_CHECK(conv_cache.dim() == 3, "conv_cache must be [N, K-1, C]");
  TORCH_CHECK(state_indices.scalar_type() == torch::kInt32,
              "state_indices must be int32");

  const int64_t B = x_bc.size(0);
  const int64_t C = x_bc.size(1);
  const int64_t K = weight_ck.size(1);
  TORCH_CHECK(weight_ck.size(0) == C, "weight_ck[0] must equal C");
  TORCH_CHECK(conv_cache.size(1) == K - 1 && conv_cache.size(2) == C,
              "conv_cache shape must be [N, K-1, C]");
  TORCH_CHECK(K >= 2 && K <= 4, "kernel supports width in [2, 4]");

  auto device = x_bc.device();
  c10::DeviceGuard guard(device);

  auto out_bc = torch::empty_like(x_bc);
  // Present x/out as [B, C, seqlen=1] to match kernel expectation.
  auto x_bcs = x_bc.unsqueeze(-1);
  auto out_bcs = out_bc.unsqueeze(-1);
  auto conv_state = conv_cache.transpose(1, 2);  // [N, C, K-1]
  auto state_idx_i32 = state_indices.contiguous();

  ConvParamsBase params;
  set_conv_params_fwd(params,
                      B,
                      C,
                      /*seqlen=*/1,
                      K,
                      x_bcs,
                      weight_ck,
                      out_bcs,
                      /*bias_ptr=*/nullptr,
                      activation,
                      /*varlen=*/false);
  params.conv_state_ptr = conv_state.data_ptr();
  params.conv_state_len = static_cast<int>(K - 1);
  params.conv_state_batch_stride = conv_state.stride(0);
  params.conv_state_c_stride = conv_state.stride(1);
  params.conv_state_l_stride = conv_state.stride(2);
  params.conv_state_indices_ptr = state_idx_i32.data_ptr<int32_t>();

  auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  DISPATCH_ITYPE(x_bc.scalar_type(), "causal_conv1d_update", [&] {
    DISPATCH_WTYPE(weight_ck.scalar_type(), "causal_conv1d_update", [&] {
      causal_conv1d_update_cuda<input_t, weight_t>(params, stream);
    });
  });
  return out_bc;
}

}  // namespace dcu
}  // namespace kernel
}  // namespace xllm
