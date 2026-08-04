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

// adapted from
// https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d_fwd.cu

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <torch/all.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>

#include "core/kernels/musa/gdn_ops.h"

namespace {

bool is_accel_device_tensor(const at::Tensor& tensor) {
  const auto dt = tensor.device().type();
  return dt == c10::DeviceType::CUDA || dt == c10::DeviceType::PrivateUse1;
}

}  // namespace

namespace xllm {
namespace kernel {
namespace musa {

namespace {

struct ConvParamsBase {
  using index_t = uint32_t;

  int batch, dim, seqlen, width;
  int64_t pad_slot_id;
  bool silu_activation;

  index_t x_batch_stride;
  index_t x_c_stride;
  index_t x_l_stride;
  index_t weight_c_stride;
  index_t weight_width_stride;
  index_t out_batch_stride;
  index_t out_c_stride;
  index_t out_l_stride;

  int conv_state_len;
  index_t conv_state_batch_stride;
  index_t conv_state_c_stride;
  index_t conv_state_l_stride;

  void* __restrict__ x_ptr;
  void* __restrict__ weight_ptr;
  void* __restrict__ bias_ptr;
  void* __restrict__ out_ptr;

  void* __restrict__ conv_state_ptr;
  void* __restrict__ query_start_loc_ptr;
  void* __restrict__ has_initial_state_ptr;
  void* __restrict__ cache_indices_ptr;
  int32_t* __restrict__ cache_seqlens;

  int32_t* __restrict__ conv_state_indices_ptr;

  void* __restrict__ seq_idx_ptr;
};

template <typename T>
__device__ inline T shuffle_xor(T val, int offset) {
  return __shfl_xor_sync(uint32_t(-1), val, offset);
}

constexpr size_t custom_max(std::initializer_list<size_t> ilist) {
  return std::max(ilist);
}

template <typename T>
constexpr T constexpr_min(T a, T b) {
  return std::min(a, b);
}

template <int BYTES>
struct BytesToType {};
template <>
struct BytesToType<16> {
  using Type = uint4;
  static_assert(sizeof(Type) == 16);
};
template <>
struct BytesToType<8> {
  using Type = uint64_t;
  static_assert(sizeof(Type) == 8);
};
template <>
struct BytesToType<4> {
  using Type = uint32_t;
  static_assert(sizeof(Type) == 4);
};
template <>
struct BytesToType<2> {
  using Type = uint16_t;
  static_assert(sizeof(Type) == 2);
};
template <>
struct BytesToType<1> {
  using Type = uint8_t;
  static_assert(sizeof(Type) == 1);
};

template <typename T>
struct SumOp {
  __device__ inline T operator()(T const& x, T const& y) { return x + y; }
};

template <int THREADS>
struct Allreduce {
  static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
  template <typename T, typename Operator>
  static __device__ inline T run(T x, Operator& op) {
    constexpr int OFFSET = THREADS / 2;
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
    return Allreduce<OFFSET>::run(x, op);
  }
};

template <>
struct Allreduce<2> {
  template <typename T, typename Operator>
  static __device__ inline T run(T x, Operator& op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
  }
};

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      static constexpr bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      static constexpr bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_WTYPE_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)     \
  if (ITYPE == at::ScalarType::Half) {                                     \
    using input_t = at::Half;                                              \
    using weight_t = at::Half;                                             \
    __VA_ARGS__();                                                         \
  } else if (ITYPE == at::ScalarType::BFloat16) {                          \
    using input_t = at::BFloat16;                                          \
    using weight_t = at::BFloat16;                                         \
    __VA_ARGS__();                                                         \
  } else if (ITYPE == at::ScalarType::Float) {                             \
    using input_t = float;                                                 \
    using weight_t = float;                                                \
    __VA_ARGS__();                                                         \
  } else {                                                                 \
    AT_ERROR(                                                              \
        #NAME, " not implemented for input type '", toString(ITYPE), "'"); \
  }

template <typename input_t, typename weight_t>
void causal_conv1d_fwd_cuda(ConvParamsBase& params, cudaStream_t stream);

void set_conv_params_fwd(
    ConvParamsBase& params,
    const size_t batch,
    const size_t dim,
    const size_t seqlen,
    const size_t width,
    const at::Tensor x,
    const at::Tensor weight,
    const at::Tensor out,
    const std::optional<at::Tensor>& bias,
    bool silu_activation,
    int64_t pad_slot_id,
    const std::optional<at::Tensor>& query_start_loc = std::nullopt,
    const std::optional<at::Tensor>& cache_indices = std::nullopt,
    const std::optional<at::Tensor>& has_initial_state = std::nullopt) {
  memset(&params, 0, sizeof(params));
  params.batch = batch;
  params.dim = dim;
  params.seqlen = seqlen;
  params.width = width;
  params.pad_slot_id = pad_slot_id;
  params.silu_activation = silu_activation;
  params.x_ptr = x.data_ptr();
  params.weight_ptr = weight.data_ptr();
  params.bias_ptr = bias.has_value() ? bias.value().data_ptr() : nullptr;
  params.out_ptr = out.data_ptr();
  params.query_start_loc_ptr = query_start_loc.has_value()
                                   ? query_start_loc.value().data_ptr()
                                   : nullptr;
  params.cache_indices_ptr =
      cache_indices.has_value() ? cache_indices.value().data_ptr() : nullptr;
  params.has_initial_state_ptr = has_initial_state.has_value()
                                     ? has_initial_state.value().data_ptr()
                                     : nullptr;
  const bool varlen = params.query_start_loc_ptr != nullptr;
  params.x_batch_stride = x.stride(varlen ? 1 : 0);
  params.x_c_stride = x.stride(varlen ? 0 : 1);
  params.x_l_stride = x.stride(varlen ? 1 : -1);
  params.weight_c_stride = weight.stride(0);
  params.weight_width_stride = weight.stride(1);
  params.out_batch_stride = out.stride(varlen ? 1 : 0);
  params.out_c_stride = out.stride(varlen ? 0 : 1);
  params.out_l_stride = out.stride(varlen ? 1 : -1);
}

template <int kNThreads_,
          int kWidth_,
          bool kIsVecLoad_,
          typename input_t_,
          typename weight_t_>
struct Causal_conv1d_fwd_kernel_traits {
  using input_t = input_t_;
  using weight_t = weight_t_;
  static constexpr int kNThreads = kNThreads_;
  static constexpr int kWidth = kWidth_;
  static constexpr int kNBytes = sizeof(input_t);
  static_assert(kNBytes == 2 || kNBytes == 4);
  static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
  static_assert(kWidth <= kNElts);
  static constexpr bool kIsVecLoad = kIsVecLoad_;
  using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
  using BlockLoadT = cub::
      BlockLoad<input_t, kNThreads, kNElts, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockLoadVecT =
      cub::BlockLoad<vec_t, kNThreads, 1, cub::BLOCK_LOAD_DIRECT>;
  using BlockStoreT = cub::
      BlockStore<input_t, kNThreads, kNElts, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  using BlockStoreVecT =
      cub::BlockStore<vec_t, kNThreads, 1, cub::BLOCK_STORE_DIRECT>;
  static constexpr int kSmemIOSize =
      kIsVecLoad ? 0
                 : custom_max({sizeof(typename BlockLoadT::TempStorage),
                               sizeof(typename BlockStoreT::TempStorage)});
  static constexpr int kSmemExchangeSize = kNThreads * kNBytes * kNElts;
  static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize;
};

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads) void causal_conv1d_fwd_kernel(
    ConvParamsBase params) {
  constexpr int kWidth = Ktraits::kWidth;
  constexpr int kNThreads = Ktraits::kNThreads;
  constexpr int kNElts = Ktraits::kNElts;
  constexpr bool kIsVecLoad = Ktraits::kIsVecLoad;
  using input_t = typename Ktraits::input_t;
  using vec_t = typename Ktraits::vec_t;
  using weight_t = typename Ktraits::weight_t;

  extern __shared__ char smem_[];
  auto& smem_load =
      reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
  auto& smem_load_vec =
      reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(smem_);
  auto& smem_store =
      reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
  auto& smem_store_vec =
      reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(smem_);
  vec_t* smem_exchange = reinterpret_cast<vec_t*>(smem_ + Ktraits::kSmemIOSize);

  const bool kVarlen = params.query_start_loc_ptr != nullptr;
  const int tidx = threadIdx.x;
  const int batch_id = blockIdx.x;
  const int channel_id = blockIdx.y;
  const int* query_start_loc =
      kVarlen ? reinterpret_cast<int*>(params.query_start_loc_ptr) : nullptr;
  const int sequence_start_index =
      kVarlen ? query_start_loc[batch_id] : batch_id;
  const int seqlen = kVarlen
                         ? query_start_loc[batch_id + 1] - sequence_start_index
                         : params.seqlen;

  input_t* x = reinterpret_cast<input_t*>(params.x_ptr) +
               sequence_start_index * params.x_batch_stride +
               channel_id * params.x_c_stride;
  weight_t* weight = reinterpret_cast<weight_t*>(params.weight_ptr) +
                     channel_id * params.weight_c_stride;
  input_t* out = reinterpret_cast<input_t*>(params.out_ptr) +
                 sequence_start_index * params.out_batch_stride +
                 channel_id * params.out_c_stride;
  float bias_val =
      params.bias_ptr == nullptr
          ? 0.f
          : float(reinterpret_cast<weight_t*>(params.bias_ptr)[channel_id]);

  bool has_initial_state =
      params.has_initial_state_ptr == nullptr
          ? false
          : reinterpret_cast<bool*>(params.has_initial_state_ptr)[batch_id];

  int* cache_indices = params.cache_indices_ptr == nullptr
                           ? nullptr
                           : reinterpret_cast<int*>(params.cache_indices_ptr);
  int cache_index =
      cache_indices == nullptr ? batch_id : cache_indices[batch_id];
  if (cache_index == params.pad_slot_id) {
    return;
  }
  input_t* conv_states =
      params.conv_state_ptr == nullptr
          ? nullptr
          : reinterpret_cast<input_t*>(params.conv_state_ptr) +
                cache_index * params.conv_state_batch_stride +
                channel_id * params.conv_state_c_stride;

  if (tidx == 0) {
    input_t initial_state[kNElts] = {0};
    if (has_initial_state) {
#pragma unroll
      for (int w = 0; w < kWidth - 1; ++w) {
        initial_state[kNElts - 1 - (kWidth - 2) + w] = conv_states[w];
      }
    }
    smem_exchange[kNThreads - 1] = reinterpret_cast<vec_t*>(initial_state)[0];
  }

  float weight_vals[kWidth];
#pragma unroll
  for (int i = 0; i < kWidth; ++i) {
    weight_vals[i] = float(weight[i * params.weight_width_stride]);
  }

  constexpr int kChunkSize = kNThreads * kNElts;
  const int n_chunks = (seqlen + kChunkSize - 1) / kChunkSize;
  for (int chunk = 0; chunk < n_chunks; ++chunk) {
    input_t x_vals_load[2 * kNElts] = {0};
    if constexpr (kIsVecLoad) {
      typename Ktraits::BlockLoadVecT(smem_load_vec)
          .Load(reinterpret_cast<vec_t*>(x),
                *reinterpret_cast<vec_t(*)[1]>(&x_vals_load[kNElts]),
                (seqlen - chunk * kChunkSize) / kNElts);
    } else {
      __syncthreads();
      typename Ktraits::BlockLoadT(smem_load).Load(
          x,
          *reinterpret_cast<input_t(*)[kNElts]>(&x_vals_load[kNElts]),
          seqlen - chunk * kChunkSize);
    }
    x += kChunkSize;
    __syncthreads();
    if (tidx < kNThreads - 1) {
      smem_exchange[tidx] = reinterpret_cast<vec_t*>(x_vals_load)[1];
    }
    __syncthreads();
    reinterpret_cast<vec_t*>(x_vals_load)[0] =
        smem_exchange[tidx > 0 ? tidx - 1 : kNThreads - 1];
    __syncthreads();
    if (tidx == kNThreads - 1) {
      smem_exchange[tidx] = reinterpret_cast<vec_t*>(x_vals_load)[1];
    }

    float x_vals[2 * kNElts];
#pragma unroll
    for (int i = 0; i < 2 * kNElts; ++i) {
      x_vals[i] = float(x_vals_load[i]);
    }

    float out_vals[kNElts];
#pragma unroll
    for (int i = 0; i < kNElts; ++i) {
      out_vals[i] = bias_val;
#pragma unroll
      for (int w = 0; w < kWidth; ++w) {
        out_vals[i] += weight_vals[w] * x_vals[kNElts + i - (kWidth - w - 1)];
      }
    }

    if (params.silu_activation) {
#pragma unroll
      for (int i = 0; i < kNElts; ++i) {
        out_vals[i] = out_vals[i] / (1 + expf(-out_vals[i]));
      }
    }

    input_t out_vals_store[kNElts];
#pragma unroll
    for (int i = 0; i < kNElts; ++i) {
      out_vals_store[i] = out_vals[i];
    }
    if constexpr (kIsVecLoad) {
      typename Ktraits::BlockStoreVecT(smem_store_vec)
          .Store(reinterpret_cast<vec_t*>(out),
                 reinterpret_cast<vec_t(&)[1]>(out_vals_store),
                 (seqlen - chunk * kChunkSize) / kNElts);
    } else {
      typename Ktraits::BlockStoreT(smem_store)
          .Store(out, out_vals_store, seqlen - chunk * kChunkSize);
    }
    out += kChunkSize;

    int final_state_position =
        ((seqlen - (kWidth - 1)) - (n_chunks - 1) * kChunkSize);
    if (conv_states != nullptr && final_state_position < 0 && seqlen > kWidth) {
      input_t vals_load[kNElts] = {0};
      if ((chunk == n_chunks - 2) && (tidx == kNThreads - 1)) {
        reinterpret_cast<vec_t*>(vals_load)[0] = smem_exchange[kNThreads - 1];
#pragma unroll
        for (int w = 0; w < -final_state_position; ++w) {
          conv_states[w] = vals_load[kNElts + final_state_position + w];
        }
      }
      if ((chunk == n_chunks - 1) && tidx == 0) {
        reinterpret_cast<vec_t*>(vals_load)[0] = smem_exchange[0];
        for (int w = -final_state_position; w < kWidth - 1; ++w) {
          conv_states[w] = vals_load[w + final_state_position];
        }
        return;
      }
    }
  }
  int last_thread =
      ((seqlen - (kWidth - 1)) - (n_chunks - 1) * kChunkSize) / kNElts;
  if (conv_states != nullptr && tidx == last_thread) {
    input_t x_vals_load[kNElts * 2] = {0};
    if (last_thread == 0 && seqlen < kWidth) {
      reinterpret_cast<vec_t*>(x_vals_load)[0] = smem_exchange[0];
      const int offset = seqlen - (kWidth - 1);
#pragma unroll
      for (int w = 0; w < kWidth - 1; ++w) {
        if ((w - seqlen) >= 0 && has_initial_state) {
          conv_states[w - seqlen] = conv_states[w];
        } else if ((w - seqlen) >= 0 && !has_initial_state) {
          conv_states[w - seqlen] = input_t(0.0f);
        }
      }
#pragma unroll
      for (int w = 0; w < kWidth - 1; ++w) {
        if (offset + w >= 0) conv_states[w] = x_vals_load[offset + w];
      }
    } else {
      const int offset = ((seqlen - (kWidth - 1)) % (kNElts));
      if ((offset + kWidth - 2) >= kNElts && (last_thread + 1 < kNThreads)) {
        reinterpret_cast<vec_t*>(x_vals_load)[1] =
            smem_exchange[last_thread + 1];
      }
      reinterpret_cast<vec_t*>(x_vals_load)[0] = smem_exchange[last_thread];
#pragma unroll
      for (int w = 0; w < kWidth - 1; ++w) {
        conv_states[w] = x_vals_load[offset + w];
      }
    }
  }
}

template <int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_fwd_launch(ConvParamsBase& params, cudaStream_t stream) {
  static constexpr int kNElts = sizeof(input_t) == 4 ? 4 : 8;
  const bool kVarlen = params.query_start_loc_ptr != nullptr;
  BOOL_SWITCH(params.seqlen % kNElts == 0 && !kVarlen, kIsVecLoad, [&] {
    using Ktraits = Causal_conv1d_fwd_kernel_traits<kNThreads,
                                                    kWidth,
                                                    kIsVecLoad,
                                                    input_t,
                                                    weight_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch, params.dim);
    auto kernel = &causal_conv1d_fwd_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
    }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

template <typename input_t, typename weight_t>
void causal_conv1d_fwd_cuda(ConvParamsBase& params, cudaStream_t stream) {
  if (params.width == 2) {
    causal_conv1d_fwd_launch<128, 2, input_t, weight_t>(params, stream);
  } else if (params.width == 3) {
    causal_conv1d_fwd_launch<128, 3, input_t, weight_t>(params, stream);
  } else if (params.width == 4) {
    causal_conv1d_fwd_launch<128, 4, input_t, weight_t>(params, stream);
  }
}

template <typename input_t, typename weight_t, int kBlockTokens>
__global__ void causal_conv1d_fwd_token_major_kernel(ConvParamsBase params) {
  const int seq_idx = blockIdx.x;
  const int token_offset = blockIdx.y * kBlockTokens;
  const int feature = blockIdx.z * blockDim.x + threadIdx.x;
  if (feature >= params.dim) {
    return;
  }

  const int32_t* query_start_loc =
      reinterpret_cast<const int32_t*>(params.query_start_loc_ptr);
  const int seq_start = query_start_loc[seq_idx];
  const int seq_len = query_start_loc[seq_idx + 1] - seq_start;
  if (token_offset >= seq_len) {
    return;
  }

  const int32_t* cache_indices =
      reinterpret_cast<const int32_t*>(params.cache_indices_ptr);
  const int cache_idx = cache_indices[seq_idx];
  if (cache_idx == params.pad_slot_id) {
    return;
  }

  const bool* has_initial_state =
      reinterpret_cast<const bool*>(params.has_initial_state_ptr);
  const bool load_initial_state = has_initial_state[seq_idx];
  const input_t* x = reinterpret_cast<const input_t*>(params.x_ptr);
  input_t* out = reinterpret_cast<input_t*>(params.out_ptr);
  const weight_t* weight =
      reinterpret_cast<const weight_t*>(params.weight_ptr) +
      feature * params.weight_c_stride;
  input_t* conv_state = reinterpret_cast<input_t*>(params.conv_state_ptr) +
                        cache_idx * params.conv_state_batch_stride +
                        feature * params.conv_state_c_stride;

  auto load_x = [&](int token_idx) {
    return static_cast<float>(x[(seq_start + token_idx) * params.x_l_stride +
                                feature * params.x_c_stride]);
  };

  float col0 = 0.0f;
  float col1 = 0.0f;
  float col2 = 0.0f;
  if (token_offset == 0) {
    if (load_initial_state) {
      col0 = static_cast<float>(conv_state[0 * params.conv_state_l_stride]);
      col1 = static_cast<float>(conv_state[1 * params.conv_state_l_stride]);
      col2 = static_cast<float>(conv_state[2 * params.conv_state_l_stride]);
    }

    if (seq_len >= 3) {
      conv_state[0 * params.conv_state_l_stride] =
          x[(seq_start + seq_len - 3) * params.x_l_stride +
            feature * params.x_c_stride];
      conv_state[1 * params.conv_state_l_stride] =
          x[(seq_start + seq_len - 2) * params.x_l_stride +
            feature * params.x_c_stride];
      conv_state[2 * params.conv_state_l_stride] =
          x[(seq_start + seq_len - 1) * params.x_l_stride +
            feature * params.x_c_stride];
    } else if (seq_len == 2) {
      conv_state[0 * params.conv_state_l_stride] =
          load_initial_state ? conv_state[2 * params.conv_state_l_stride]
                             : input_t(0.0f);
      conv_state[1 * params.conv_state_l_stride] =
          x[seq_start * params.x_l_stride + feature * params.x_c_stride];
      conv_state[2 * params.conv_state_l_stride] =
          x[(seq_start + 1) * params.x_l_stride + feature * params.x_c_stride];
    } else {
      const input_t previous1 = conv_state[1 * params.conv_state_l_stride];
      const input_t previous2 = conv_state[2 * params.conv_state_l_stride];
      conv_state[0 * params.conv_state_l_stride] =
          load_initial_state ? previous1 : input_t(0.0f);
      conv_state[1 * params.conv_state_l_stride] =
          load_initial_state ? previous2 : input_t(0.0f);
      conv_state[2 * params.conv_state_l_stride] =
          x[seq_start * params.x_l_stride + feature * params.x_c_stride];
    }
  } else {
    col0 = load_x(token_offset - 3);
    col1 = load_x(token_offset - 2);
    col2 = load_x(token_offset - 1);
  }

  const float w0 = static_cast<float>(weight[0 * params.weight_width_stride]);
  const float w1 = static_cast<float>(weight[1 * params.weight_width_stride]);
  const float w2 = static_cast<float>(weight[2 * params.weight_width_stride]);
  const float w3 = static_cast<float>(weight[3 * params.weight_width_stride]);
  const float bias = params.bias_ptr == nullptr
                         ? 0.0f
                         : static_cast<float>(reinterpret_cast<const weight_t*>(
                               params.bias_ptr)[feature]);
  const int segment_len = min(kBlockTokens, seq_len - token_offset);
#pragma unroll
  for (int token_i = 0; token_i < kBlockTokens; ++token_i) {
    if (token_i < segment_len) {
      const float current = load_x(token_offset + token_i);
      float value = bias + col0 * w0 + col1 * w1 + col2 * w2 + current * w3;
      col0 = col1;
      col1 = col2;
      col2 = current;
      if (params.silu_activation) {
        constexpr float kLog2e = 1.4426950408889634f;
        value = value / (1.0f + exp2f(-value * kLog2e));
      }
      out[(seq_start + token_offset + token_i) * params.out_l_stride +
          feature * params.out_c_stride] = input_t(value);
    }
  }
}

template <typename input_t, typename weight_t>
void causal_conv1d_fwd_token_major_cuda(ConvParamsBase& params,
                                        cudaStream_t stream) {
  constexpr int kBlockTokens = 8;
  constexpr int kThreads = 256;
  const dim3 grid(params.batch,
                  (params.seqlen + kBlockTokens - 1) / kBlockTokens,
                  (params.dim + kThreads - 1) / kThreads);
  causal_conv1d_fwd_token_major_kernel<input_t, weight_t, kBlockTokens>
      <<<grid, kThreads, 0, stream>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void causal_conv1d_fwd_cuda<float, float>(ConvParamsBase& params,
                                                   cudaStream_t stream);
template void causal_conv1d_fwd_cuda<at::Half, at::Half>(ConvParamsBase& params,
                                                         cudaStream_t stream);
template void causal_conv1d_fwd_cuda<at::BFloat16, at::BFloat16>(
    ConvParamsBase& params,
    cudaStream_t stream);

}  // namespace

void causal_conv1d_fwd(const at::Tensor& x,
                       const at::Tensor& weight,
                       at::Tensor& out,
                       const std::optional<at::Tensor>& bias_,
                       const std::optional<at::Tensor>& conv_states,
                       const std::optional<at::Tensor>& query_start_loc,
                       const std::optional<at::Tensor>& cache_indices,
                       const std::optional<at::Tensor>& has_initial_state,
                       bool silu_activation,
                       int64_t pad_slot_id) {
  auto input_type = x.scalar_type();
  auto weight_type = weight.scalar_type();
  TORCH_CHECK(input_type == at::ScalarType::Float ||
              input_type == at::ScalarType::Half ||
              input_type == at::ScalarType::BFloat16);
  TORCH_CHECK(weight_type == at::ScalarType::Float ||
              weight_type == at::ScalarType::Half ||
              weight_type == at::ScalarType::BFloat16);

  TORCH_CHECK(is_accel_device_tensor(x), "x must be a device tensor");
  TORCH_CHECK(is_accel_device_tensor(weight), "weight must be a device tensor");

  const bool varlen = query_start_loc.has_value() ? true : false;
  const auto sizes = x.sizes();
  const int batch_size =
      varlen ? query_start_loc.value().sizes()[0] - 1 : sizes[0];
  const int dim = varlen ? sizes[0] : sizes[1];
  const int seqlen = varlen ? sizes[1] : sizes[2];
  const int width = weight.size(-1);
  if (varlen) {
    CHECK_SHAPE(x, dim, seqlen);
  } else {
    CHECK_SHAPE(x, batch_size, dim, seqlen);
  }
  CHECK_SHAPE(weight, dim, width);

  if (bias_.has_value()) {
    auto bias = bias_.value();
    TORCH_CHECK(bias.scalar_type() == weight_type);
    TORCH_CHECK(is_accel_device_tensor(bias), "bias must be a device tensor");
    TORCH_CHECK(bias.stride(-1) == 1);
    CHECK_SHAPE(bias, dim);
  }

  if (has_initial_state.has_value()) {
    auto has_initial_state_ = has_initial_state.value();
    TORCH_CHECK(has_initial_state_.scalar_type() == at::ScalarType::Bool);
    TORCH_CHECK(is_accel_device_tensor(has_initial_state_),
                "has_initial_state must be a device tensor");
    CHECK_SHAPE(has_initial_state_, batch_size);
  }

  if (query_start_loc.has_value()) {
    auto query_start_loc_ = query_start_loc.value();
    TORCH_CHECK(query_start_loc_.scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(is_accel_device_tensor(query_start_loc_),
                "query_start_loc must be a device tensor");
  }

  if (cache_indices.has_value()) {
    auto cache_indices_ = cache_indices.value();
    TORCH_CHECK(cache_indices_.scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(is_accel_device_tensor(cache_indices_),
                "cache_indices must be a device tensor");
    CHECK_SHAPE(cache_indices_, batch_size);
  }

  ConvParamsBase params;
  set_conv_params_fwd(params,
                      batch_size,
                      dim,
                      seqlen,
                      width,
                      x,
                      weight,
                      out,
                      bias_,
                      silu_activation,
                      pad_slot_id,
                      query_start_loc,
                      cache_indices,
                      has_initial_state);

  if (conv_states.has_value()) {
    auto conv_states_ = conv_states.value();
    TORCH_CHECK(conv_states_.scalar_type() == input_type);
    TORCH_CHECK(is_accel_device_tensor(conv_states_),
                "conv_states must be a device tensor");
    params.conv_state_ptr = conv_states_.data_ptr();
    params.conv_state_batch_stride = conv_states_.stride(0);
    params.conv_state_c_stride = conv_states_.stride(-2);
    params.conv_state_l_stride = conv_states_.stride(-1);
  } else {
    params.conv_state_ptr = nullptr;
  }

  at::cuda::CUDAGuard device_guard{(char)x.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  DISPATCH_WTYPE_ITYPE_FLOAT_AND_HALF_AND_BF16(
      x.scalar_type(), "causal_conv1d_fwd", [&] {
        causal_conv1d_fwd_cuda<input_t, weight_t>(params, stream);
      });
}

void causal_conv1d_fwd_token_major(const at::Tensor& x,
                                   const at::Tensor& weight,
                                   at::Tensor& out,
                                   const std::optional<at::Tensor>& bias,
                                   const at::Tensor& conv_states,
                                   const at::Tensor& query_start_loc,
                                   const at::Tensor& cache_indices,
                                   const at::Tensor& has_initial_state,
                                   bool silu_activation,
                                   int64_t pad_slot_id) {
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous());
  TORCH_CHECK(out.sizes() == x.sizes() && out.is_contiguous());
  TORCH_CHECK(weight.dim() == 2 && weight.size(1) == 4);
  TORCH_CHECK(weight.size(0) == x.size(1));
  TORCH_CHECK(query_start_loc.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(cache_indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(has_initial_state.scalar_type() == at::ScalarType::Bool);

  ConvParamsBase params;
  memset(&params, 0, sizeof(params));
  params.batch = query_start_loc.size(0) - 1;
  params.dim = x.size(1);
  params.seqlen = x.size(0);
  params.width = 4;
  params.pad_slot_id = pad_slot_id;
  params.silu_activation = silu_activation;
  params.x_ptr = x.data_ptr();
  params.weight_ptr = weight.data_ptr();
  params.bias_ptr = bias.has_value() ? bias.value().data_ptr() : nullptr;
  params.out_ptr = out.data_ptr();
  params.query_start_loc_ptr = query_start_loc.data_ptr();
  params.cache_indices_ptr = cache_indices.data_ptr();
  params.has_initial_state_ptr = has_initial_state.data_ptr();
  params.x_c_stride = x.stride(1);
  params.x_l_stride = x.stride(0);
  params.weight_c_stride = weight.stride(0);
  params.weight_width_stride = weight.stride(1);
  params.out_c_stride = out.stride(1);
  params.out_l_stride = out.stride(0);
  params.conv_state_ptr = conv_states.data_ptr();
  params.conv_state_batch_stride = conv_states.stride(0);
  params.conv_state_c_stride = conv_states.stride(-2);
  params.conv_state_l_stride = conv_states.stride(-1);

  at::cuda::CUDAGuard device_guard{static_cast<char>(x.get_device())};
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  DISPATCH_WTYPE_ITYPE_FLOAT_AND_HALF_AND_BF16(
      x.scalar_type(), "causal_conv1d_fwd_token_major", [&] {
        causal_conv1d_fwd_token_major_cuda<input_t, weight_t>(params, stream);
      });
}

}  // namespace musa
}  // namespace kernel
}  // namespace xllm
