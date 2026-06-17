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

#include <ATen/hip/HIPContextLight.h>
#include <c10/hip/HIPStream.h>
#include <glog/logging.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "core/kernels/dcu/dcu_ops_api.h"

namespace xllm::kernel::dcu {

namespace {

__device__ __forceinline__ int8_t quantize_to_int8(float val,
                                                   float inv_scale) {
  return static_cast<int8_t>(
      fmaxf(-127.0f, fminf(127.0f, nearbyintf(val * inv_scale))));
}

__device__ __forceinline__ uint32_t pack_int8x4(int8_t x0,
                                                int8_t x1,
                                                int8_t x2,
                                                int8_t x3) {
  return static_cast<uint32_t>(static_cast<uint8_t>(x0)) |
         (static_cast<uint32_t>(static_cast<uint8_t>(x1)) << 8) |
         (static_cast<uint32_t>(static_cast<uint8_t>(x2)) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(x3)) << 24);
}

// Each DCU block handles one row (token). Two passes over the row:
//   Pass 1: compute per-row absmax
//   Pass 2: quantize to INT8 using the computed scale
template <typename ScalarT, int32_t BlockSize, int32_t KSize = 0>
__global__ void __launch_bounds__(BlockSize)
per_token_int8_quantize_kernel(
    const ScalarT* __restrict__ x,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t M,
    int32_t K) {
  const int32_t row_size = (KSize > 0) ? KSize : K;
  const int64_t row = blockIdx.x;
  if (row >= M) {
    return;
  }

  const int64_t row_offset = row * static_cast<int64_t>(row_size);
  const ScalarT* x_row = x + row_offset;
  int8_t* out_row = output + row_offset;

  float thread_max = 0.0f;
  for (int32_t col = threadIdx.x; col < row_size; col += BlockSize) {
    const float val = static_cast<float>(x_row[col]);
    thread_max = fmaxf(thread_max, fabsf(val));
  }

  constexpr int32_t kDcuWarpSize = 64;
  for (int32_t mask = kDcuWarpSize / 2; mask > 0; mask >>= 1) {
    thread_max = fmaxf(thread_max, __shfl_xor(thread_max, mask));
  }

  constexpr int32_t kMaxWarps = BlockSize / kDcuWarpSize;
  __shared__ float smem[kMaxWarps + 1];
  const int32_t warp_id = threadIdx.x / kDcuWarpSize;
  const int32_t lane_id = threadIdx.x % kDcuWarpSize;

  if (lane_id == 0) {
    smem[warp_id] = thread_max;
  }
  __syncthreads();

  if (warp_id == 0) {
    float v = (lane_id < kMaxWarps) ? smem[lane_id] : 0.0f;
    for (int32_t mask = kDcuWarpSize / 2; mask > 0; mask >>= 1) {
      v = fmaxf(v, __shfl_xor(v, mask));
    }
    if (lane_id == 0) {
      smem[kMaxWarps] = v;
    }
  }
  __syncthreads();

  const float row_max = smem[kMaxWarps];
  const float inv_scale = (row_max > 1e-10f) ? 127.0f / row_max : 0.0f;

  for (int32_t col = threadIdx.x; col < row_size; col += BlockSize) {
    const float val = static_cast<float>(x_row[col]);
    out_row[col] = quantize_to_int8(val, inv_scale);
  }

  if (threadIdx.x == 0) {
    scales[row] = row_max / 127.0f;
  }
}

template <typename ScalarT, int32_t BlockSize, int32_t KSize>
__global__ void __launch_bounds__(BlockSize)
per_token_int8_quantize_vec_kernel(
    const ScalarT* __restrict__ x,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t M) {
  constexpr int32_t kVecSize = 16;
  constexpr int32_t kVecCount = KSize / kVecSize;
  const int64_t row = blockIdx.x;
  if (row >= M) {
    return;
  }

  const int64_t row_offset = row * static_cast<int64_t>(KSize);
  const ScalarT* x_row = x + row_offset;
  int8_t* out_row = output + row_offset;

  float thread_max = 0.0f;
  for (int32_t vec = threadIdx.x; vec < kVecCount; vec += BlockSize) {
    const int32_t col = vec * kVecSize;
    const float v0 = static_cast<float>(x_row[col]);
    const float v1 = static_cast<float>(x_row[col + 1]);
    const float v2 = static_cast<float>(x_row[col + 2]);
    const float v3 = static_cast<float>(x_row[col + 3]);
    const float v4 = static_cast<float>(x_row[col + 4]);
    const float v5 = static_cast<float>(x_row[col + 5]);
    const float v6 = static_cast<float>(x_row[col + 6]);
    const float v7 = static_cast<float>(x_row[col + 7]);
    const float v8 = static_cast<float>(x_row[col + 8]);
    const float v9 = static_cast<float>(x_row[col + 9]);
    const float v10 = static_cast<float>(x_row[col + 10]);
    const float v11 = static_cast<float>(x_row[col + 11]);
    const float v12 = static_cast<float>(x_row[col + 12]);
    const float v13 = static_cast<float>(x_row[col + 13]);
    const float v14 = static_cast<float>(x_row[col + 14]);
    const float v15 = static_cast<float>(x_row[col + 15]);
    thread_max = fmaxf(thread_max, fabsf(v0));
    thread_max = fmaxf(thread_max, fabsf(v1));
    thread_max = fmaxf(thread_max, fabsf(v2));
    thread_max = fmaxf(thread_max, fabsf(v3));
    thread_max = fmaxf(thread_max, fabsf(v4));
    thread_max = fmaxf(thread_max, fabsf(v5));
    thread_max = fmaxf(thread_max, fabsf(v6));
    thread_max = fmaxf(thread_max, fabsf(v7));
    thread_max = fmaxf(thread_max, fabsf(v8));
    thread_max = fmaxf(thread_max, fabsf(v9));
    thread_max = fmaxf(thread_max, fabsf(v10));
    thread_max = fmaxf(thread_max, fabsf(v11));
    thread_max = fmaxf(thread_max, fabsf(v12));
    thread_max = fmaxf(thread_max, fabsf(v13));
    thread_max = fmaxf(thread_max, fabsf(v14));
    thread_max = fmaxf(thread_max, fabsf(v15));
  }

  constexpr int32_t kDcuWarpSize = 64;
  for (int32_t mask = kDcuWarpSize / 2; mask > 0; mask >>= 1) {
    thread_max = fmaxf(thread_max, __shfl_xor(thread_max, mask));
  }

  constexpr int32_t kMaxWarps = BlockSize / kDcuWarpSize;
  __shared__ float smem[kMaxWarps + 1];
  const int32_t warp_id = threadIdx.x / kDcuWarpSize;
  const int32_t lane_id = threadIdx.x % kDcuWarpSize;

  if (lane_id == 0) {
    smem[warp_id] = thread_max;
  }
  __syncthreads();

  if (warp_id == 0) {
    float v = (lane_id < kMaxWarps) ? smem[lane_id] : 0.0f;
    for (int32_t mask = kDcuWarpSize / 2; mask > 0; mask >>= 1) {
      v = fmaxf(v, __shfl_xor(v, mask));
    }
    if (lane_id == 0) {
      smem[kMaxWarps] = v;
    }
  }
  __syncthreads();

  const float row_max = smem[kMaxWarps];
  const float inv_scale = (row_max > 1e-10f) ? 127.0f / row_max : 0.0f;
  uint32_t* out_vec = reinterpret_cast<uint32_t*>(out_row);

  for (int32_t vec = threadIdx.x; vec < kVecCount; vec += BlockSize) {
    const int32_t col = vec * kVecSize;
    const float v0 = static_cast<float>(x_row[col]);
    const float v1 = static_cast<float>(x_row[col + 1]);
    const float v2 = static_cast<float>(x_row[col + 2]);
    const float v3 = static_cast<float>(x_row[col + 3]);
    const float v4 = static_cast<float>(x_row[col + 4]);
    const float v5 = static_cast<float>(x_row[col + 5]);
    const float v6 = static_cast<float>(x_row[col + 6]);
    const float v7 = static_cast<float>(x_row[col + 7]);
    const float v8 = static_cast<float>(x_row[col + 8]);
    const float v9 = static_cast<float>(x_row[col + 9]);
    const float v10 = static_cast<float>(x_row[col + 10]);
    const float v11 = static_cast<float>(x_row[col + 11]);
    const float v12 = static_cast<float>(x_row[col + 12]);
    const float v13 = static_cast<float>(x_row[col + 13]);
    const float v14 = static_cast<float>(x_row[col + 14]);
    const float v15 = static_cast<float>(x_row[col + 15]);
    const int32_t out_col = vec * 4;
    out_vec[out_col] = pack_int8x4(quantize_to_int8(v0, inv_scale),
                                   quantize_to_int8(v1, inv_scale),
                                   quantize_to_int8(v2, inv_scale),
                                   quantize_to_int8(v3, inv_scale));
    out_vec[out_col + 1] = pack_int8x4(quantize_to_int8(v4, inv_scale),
                                       quantize_to_int8(v5, inv_scale),
                                       quantize_to_int8(v6, inv_scale),
                                       quantize_to_int8(v7, inv_scale));
    out_vec[out_col + 2] = pack_int8x4(quantize_to_int8(v8, inv_scale),
                                       quantize_to_int8(v9, inv_scale),
                                       quantize_to_int8(v10, inv_scale),
                                       quantize_to_int8(v11, inv_scale));
    out_vec[out_col + 3] = pack_int8x4(quantize_to_int8(v12, inv_scale),
                                       quantize_to_int8(v13, inv_scale),
                                       quantize_to_int8(v14, inv_scale),
                                       quantize_to_int8(v15, inv_scale));
  }

  if (threadIdx.x == 0) {
    scales[row] = row_max / 127.0f;
  }
}

int32_t select_specialized_k(int32_t k) {
  if (k == 2560 || k == 4096 || k == 5120 || k == 7168) {
    return k;
  }
  return 0;
}

bool use_vectorized_kernel(int64_t m, int32_t k) {
  return select_specialized_k(k) != 0 && m >= 512;
}

int32_t select_block_size(int64_t m, int64_t k) {
  if (m <= 16 || (k >= 5120 && m <= 128)) {
    return 1024;
  }
  if (k >= 5120 && m >= 1024 && m < 4096) {
    return 1024;
  }
  if (m >= 256 && m <= 1024) {
    return 256;
  }
  return 512;
}

int32_t select_vectorized_block_size(int64_t m, int64_t k) {
  if (k >= 7168 && m >= 8192) {
    return 1024;
  }
  return select_block_size(m, k);
}

template <typename ScalarT, int32_t BlockSize, int32_t KSize>
void launch_per_token_int8_quantize_vec_kernel(const torch::Tensor& x,
                                               torch::Tensor& x_q,
                                               torch::Tensor& out_scales,
                                               int64_t M,
                                               hipStream_t stream) {
  per_token_int8_quantize_vec_kernel<ScalarT, BlockSize, KSize>
      <<<M, BlockSize, 0, stream>>>(x.data_ptr<ScalarT>(),
                                     x_q.data_ptr<int8_t>(),
                                     out_scales.data_ptr<float>(),
                                     M);
}

template <typename ScalarT, int32_t BlockSize, int32_t KSize>
void launch_per_token_int8_quantize_kernel(const torch::Tensor& x,
                                           torch::Tensor& x_q,
                                           torch::Tensor& out_scales,
                                           int64_t M,
                                           int32_t K,
                                           hipStream_t stream) {
  per_token_int8_quantize_kernel<ScalarT, BlockSize, KSize><<<M,
                                                                 BlockSize,
                                                                 0,
                                                                 stream>>>(
      x.data_ptr<ScalarT>(),
      x_q.data_ptr<int8_t>(),
      out_scales.data_ptr<float>(),
      M,
      K);
}

template <typename ScalarT, int32_t BlockSize>
bool launch_specialized_vec_kernel(const torch::Tensor& x,
                                   torch::Tensor& x_q,
                                   torch::Tensor& out_scales,
                                   int64_t M,
                                   int32_t K,
                                   hipStream_t stream) {
  if (K == 2560) {
    launch_per_token_int8_quantize_vec_kernel<ScalarT, BlockSize, 2560>(
        x, x_q, out_scales, M, stream);
    return true;
  }
  if (K == 4096) {
    launch_per_token_int8_quantize_vec_kernel<ScalarT, BlockSize, 4096>(
        x, x_q, out_scales, M, stream);
    return true;
  }
  if (K == 5120) {
    launch_per_token_int8_quantize_vec_kernel<ScalarT, BlockSize, 5120>(
        x, x_q, out_scales, M, stream);
    return true;
  }
  if (K == 7168) {
    launch_per_token_int8_quantize_vec_kernel<ScalarT, BlockSize, 7168>(
        x, x_q, out_scales, M, stream);
    return true;
  }
  return false;
}

template <typename ScalarT, int32_t BlockSize>
bool launch_specialized_scalar_kernel(const torch::Tensor& x,
                                      torch::Tensor& x_q,
                                      torch::Tensor& out_scales,
                                      int64_t M,
                                      int32_t K,
                                      hipStream_t stream) {
  if (K == 2560) {
    launch_per_token_int8_quantize_kernel<ScalarT, BlockSize, 2560>(
        x, x_q, out_scales, M, K, stream);
    return true;
  }
  if (K == 4096) {
    launch_per_token_int8_quantize_kernel<ScalarT, BlockSize, 4096>(
        x, x_q, out_scales, M, K, stream);
    return true;
  }
  if (K == 5120) {
    launch_per_token_int8_quantize_kernel<ScalarT, BlockSize, 5120>(
        x, x_q, out_scales, M, K, stream);
    return true;
  }
  if (K == 7168) {
    launch_per_token_int8_quantize_kernel<ScalarT, BlockSize, 7168>(
        x, x_q, out_scales, M, K, stream);
    return true;
  }
  return false;
}

template <typename ScalarT>
void dispatch_per_token_int8_quantize_kernel(const torch::Tensor& x,
                                             torch::Tensor& x_q,
                                             torch::Tensor& out_scales,
                                             int64_t M,
                                             int32_t K,
                                             hipStream_t stream) {
  const int32_t specialized_k = select_specialized_k(K);
  if (use_vectorized_kernel(M, K)) {
    const int32_t block_size = select_vectorized_block_size(M, K);
    if (block_size == 256 &&
        launch_specialized_vec_kernel<ScalarT, 256>(
            x, x_q, out_scales, M, specialized_k, stream)) {
      return;
    }
    if (block_size == 512 &&
        launch_specialized_vec_kernel<ScalarT, 512>(
            x, x_q, out_scales, M, specialized_k, stream)) {
      return;
    }
    if (block_size == 1024 &&
        launch_specialized_vec_kernel<ScalarT, 1024>(
            x, x_q, out_scales, M, specialized_k, stream)) {
      return;
    }
  }
  const int32_t block_size = select_block_size(M, K);
  if (block_size == 256) {
    launch_per_token_int8_quantize_kernel<ScalarT, 256, 0>(
        x, x_q, out_scales, M, K, stream);
    return;
  }
  if (block_size == 1024) {
    if (launch_specialized_scalar_kernel<ScalarT, 1024>(
            x, x_q, out_scales, M, specialized_k, stream)) {
      return;
    }
    launch_per_token_int8_quantize_kernel<ScalarT, 1024, 0>(
        x, x_q, out_scales, M, K, stream);
    return;
  }
  if (launch_specialized_scalar_kernel<ScalarT, 512>(
          x, x_q, out_scales, M, specialized_k, stream)) {
    return;
  }
  launch_per_token_int8_quantize_kernel<ScalarT, 512, 0>(
      x, x_q, out_scales, M, K, stream);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> scaled_quantize(
    const torch::Tensor& x,
    const torch::Tensor& smooth,
    const std::optional<torch::Tensor>& zero,
    const std::optional<torch::Tensor>& token_count,
    const std::optional<torch::Tensor>& gather_index,
    const std::optional<torch::Tensor>& gather_index_start_position,
    const std::optional<torch::Tensor>& output,
    const std::optional<torch::Tensor>& output_scale,
    const std::string& act_mode,
    double /*active_coef*/,
    bool is_gated,
    at::ScalarType quant_type) {
  CHECK(!smooth.defined() || smooth.numel() == 0)
      << "DCU scaled_quantize only supports no-smooth per-token int8 quant.";
  CHECK(!zero.has_value())
      << "DCU scaled_quantize does not support zero point.";
  CHECK(!token_count.has_value())
      << "DCU scaled_quantize does not support token_count.";
  CHECK(!gather_index.has_value())
      << "DCU scaled_quantize does not support gather_index.";
  CHECK(!gather_index_start_position.has_value())
      << "DCU scaled_quantize does not support gather_index_start_position.";
  CHECK(!output.has_value()) << "DCU scaled_quantize does not support output.";
  CHECK(!output_scale.has_value())
      << "DCU scaled_quantize does not support output_scale.";
  CHECK(act_mode == "none")
      << "DCU scaled_quantize only supports act_mode=none.";
  CHECK(!is_gated) << "DCU scaled_quantize does not support gated activation.";
  CHECK(quant_type == torch::kChar)
      << "DCU scaled_quantize only supports int8 output.";
  CHECK(x.dim() == 2) << "DCU scaled_quantize expects a 2D input tensor.";
  CHECK(x.is_contiguous()) << "DCU scaled_quantize expects contiguous input.";

  const int64_t M = x.size(0);
  const int32_t K = static_cast<int32_t>(x.size(-1));

  torch::Tensor x_q = torch::empty(
      {M, K}, torch::TensorOptions().dtype(torch::kChar).device(x.device()));
  torch::Tensor out_scales = torch::empty(
      {M},
      torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

  hipStream_t stream = c10::hip::getCurrentHIPStream();

  if (x.scalar_type() == torch::kHalf) {
    dispatch_per_token_int8_quantize_kernel<at::Half>(
        x, x_q, out_scales, M, K, stream);
  } else {
    CHECK(x.scalar_type() == torch::kBFloat16)
        << "scaled_quantize only supports half or bfloat16 input";
    dispatch_per_token_int8_quantize_kernel<at::BFloat16>(
        x, x_q, out_scales, M, K, stream);
  }

  return std::make_tuple(x_q, out_scales);
}

}  // namespace xllm::kernel::dcu
