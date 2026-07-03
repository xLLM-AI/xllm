/* Copyright 2025 The vLLM Authors and The xLLM Authors. All Rights Reserved.

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
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "core/kernels/cuda/device_utils.cuh"
#include "core/kernels/musa/musa_ops_api.h"

namespace {

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr,
    const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr,
    int rot_offset,
    int embed_dim) {
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = *(cos_ptr + x_index);
    sin = *(sin_ptr + x_index);
  } else {
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = *(cos_ptr + x_index / 2);
    sin = *(sin_ptr + x_index / 2);
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(scalar_t* __restrict__ query,
                                              scalar_t* __restrict__ key,
                                              const scalar_t* cache_ptr,
                                              const int head_size,
                                              const int num_heads,
                                              const int num_kv_heads,
                                              const int rot_dim,
                                              const int token_idx,
                                              const int64_t query_stride,
                                              const int64_t key_stride,
                                              const int64_t head_stride) {
  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head =
        token_idx * query_stride + head_idx * head_stride;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, IS_NEOX>(
        query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }

  if (key != nullptr) {
    const int nk = num_kv_heads * embed_dim;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
      const int head_idx = i / embed_dim;
      const int64_t token_head =
          token_idx * key_stride + head_idx * head_stride;
      const int rot_offset = i % embed_dim;
      apply_token_rotary_embedding<scalar_t, IS_NEOX>(
          key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
    }
  }
}

template <typename scalar_t, typename idx_t, bool IS_NEOX>
__global__ void XLLM_KERNEL_ATTR(512)
    rotary_embedding_kernel(const idx_t* __restrict__ positions,
                            scalar_t* __restrict__ query,
                            scalar_t* __restrict__ key,
                            const scalar_t* __restrict__ cos_sin_cache,
                            const int rot_dim,
                            const int64_t query_stride,
                            const int64_t key_stride,
                            const int64_t head_stride,
                            const int num_heads,
                            const int num_kv_heads,
                            const int head_size) {
  const int token_idx = blockIdx.x;
  int64_t pos = static_cast<int64_t>(positions[token_idx]);
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_rotary_embedding<scalar_t, IS_NEOX>(query,
                                            key,
                                            cache_ptr,
                                            head_size,
                                            num_heads,
                                            num_kv_heads,
                                            rot_dim,
                                            token_idx,
                                            query_stride,
                                            key_stride,
                                            head_stride);
}
}  // namespace

namespace xllm::kernel::cuda {

void rotary_embedding(torch::Tensor& positions,
                      torch::Tensor& query,
                      std::optional<torch::Tensor> key,
                      torch::Tensor& cos_sin_cache,
                      bool is_neox) {
  int64_t head_size = cos_sin_cache.size(-1);
  int64_t num_tokens = positions.numel();
  int positions_ndim = positions.dim();

  CHECK(positions_ndim == 1 || positions_ndim == 2)
      << "positions must have shape [num_tokens] or [batch_size, seq_len]";

  if (positions_ndim == 1) {
    CHECK(query.size(0) == positions.size(0) &&
          (!key.has_value() || key->size(0) == positions.size(0)))
        << "query, key and positions must have the same number of tokens";
  }
  if (positions_ndim == 2) {
    CHECK(query.size(0) == positions.size(0) &&
          (!key.has_value() || key->size(0) == positions.size(0)) &&
          query.size(1) == positions.size(1) &&
          (!key.has_value() || key->size(1) == positions.size(1)))
        << "query, key and positions must have the same batch_size and seq_len";
  }

  int query_hidden_size = query.numel() / num_tokens;
  int key_hidden_size = key.has_value() ? key->numel() / num_tokens : 0;
  CHECK(query_hidden_size % head_size == 0);
  CHECK(key_hidden_size % head_size == 0);

  int num_heads = query_hidden_size / head_size;
  int num_kv_heads = key.has_value() ? key_hidden_size / head_size : num_heads;
  CHECK(num_heads % num_kv_heads == 0);

  int rot_dim = cos_sin_cache.size(1);
  int seq_dim_idx = positions_ndim - 1;
  int64_t query_stride = query.stride(seq_dim_idx);
  int64_t key_stride = key.has_value() ? key->stride(seq_dim_idx) : 0;
  int query_ndim = query.dim();
  int64_t head_stride =
      (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;

  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  CHECK(positions.scalar_type() == torch::kInt32 ||
        positions.scalar_type() == torch::kInt64)
      << "positions must be int32 or int64, got " << positions.scalar_type();
  const bool positions_is_int64 = positions.scalar_type() == torch::kInt64;
  DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "apply_rope_pos_ids_cos_sin_cache", [&] {
        scalar_t* query_ptr = query.data_ptr<scalar_t>();
        scalar_t* key_ptr =
            key.has_value() ? key->data_ptr<scalar_t>() : nullptr;
        const scalar_t* cache_ptr = cos_sin_cache.data_ptr<scalar_t>();
        if (positions_is_int64) {
          const int64_t* pos_ptr = positions.data_ptr<int64_t>();
          if (is_neox) {
            rotary_embedding_kernel<scalar_t, int64_t, true>
                <<<grid, block, 0, stream>>>(pos_ptr,
                                             query_ptr,
                                             key_ptr,
                                             cache_ptr,
                                             rot_dim,
                                             query_stride,
                                             key_stride,
                                             head_stride,
                                             num_heads,
                                             num_kv_heads,
                                             head_size);
          } else {
            rotary_embedding_kernel<scalar_t, int64_t, false>
                <<<grid, block, 0, stream>>>(pos_ptr,
                                             query_ptr,
                                             key_ptr,
                                             cache_ptr,
                                             rot_dim,
                                             query_stride,
                                             key_stride,
                                             head_stride,
                                             num_heads,
                                             num_kv_heads,
                                             head_size);
          }
        } else {
          const int32_t* pos_ptr = positions.data_ptr<int32_t>();
          if (is_neox) {
            rotary_embedding_kernel<scalar_t, int32_t, true>
                <<<grid, block, 0, stream>>>(pos_ptr,
                                             query_ptr,
                                             key_ptr,
                                             cache_ptr,
                                             rot_dim,
                                             query_stride,
                                             key_stride,
                                             head_stride,
                                             num_heads,
                                             num_kv_heads,
                                             head_size);
          } else {
            rotary_embedding_kernel<scalar_t, int32_t, false>
                <<<grid, block, 0, stream>>>(pos_ptr,
                                             query_ptr,
                                             key_ptr,
                                             cache_ptr,
                                             rot_dim,
                                             query_stride,
                                             key_stride,
                                             head_stride,
                                             num_heads,
                                             num_kv_heads,
                                             head_size);
          }
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void partial_rotary_embedding_inplace(torch::Tensor& positions,
                                      torch::Tensor& query,
                                      torch::Tensor& key,
                                      torch::Tensor& cos_sin_cache,
                                      int64_t head_size,
                                      int64_t rotary_dim,
                                      bool is_neox) {
  CHECK(head_size > 0)
      << "partial_rotary_embedding_inplace: head_size must be > 0";
  CHECK(rotary_dim > 0 && rotary_dim <= head_size)
      << "partial_rotary_embedding_inplace: 0 < rotary_dim <= head_size";
  CHECK(rotary_dim % 2 == 0)
      << "partial_rotary_embedding_inplace: rotary_dim must be even";
  CHECK(cos_sin_cache.size(-1) == rotary_dim)
      << "partial_rotary_embedding_inplace: cos_sin_cache last dim ("
      << cos_sin_cache.size(-1) << ") must equal rotary_dim (" << rotary_dim
      << ")";

  const int64_t num_tokens = positions.numel();
  const int32_t positions_ndim = static_cast<int32_t>(positions.dim());
  CHECK(positions_ndim == 1 || positions_ndim == 2)
      << "positions must have shape [num_tokens] or [batch_size, seq_len]";

  const int64_t query_hidden_size = query.numel() / num_tokens;
  const int64_t key_hidden_size = key.numel() / num_tokens;
  CHECK(query_hidden_size % head_size == 0)
      << "partial_rotary_embedding_inplace: query hidden_size must be "
         "divisible by head_size";
  CHECK(key_hidden_size % head_size == 0)
      << "partial_rotary_embedding_inplace: key hidden_size must be "
         "divisible by head_size";
  CHECK(query.stride(-1) == 1 && key.stride(-1) == 1)
      << "partial_rotary_embedding_inplace: query/key last dim must be "
         "contiguous (stride==1)";
  CHECK(cos_sin_cache.is_contiguous())
      << "partial_rotary_embedding_inplace: cos_sin_cache must be contiguous";

  const int32_t num_heads = static_cast<int32_t>(query_hidden_size / head_size);
  const int32_t num_kv_heads =
      static_cast<int32_t>(key_hidden_size / head_size);
  CHECK(num_kv_heads > 0 && num_heads % num_kv_heads == 0)
      << "partial_rotary_embedding_inplace: num_heads must be a multiple "
         "of num_kv_heads";

  const int32_t seq_dim_idx = positions_ndim - 1;
  const int64_t query_stride = query.stride(seq_dim_idx);
  const int64_t key_stride = key.stride(seq_dim_idx);
  const int32_t query_ndim = static_cast<int32_t>(query.dim());
  const int64_t head_stride =
      (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;

  dim3 grid(static_cast<unsigned>(num_tokens));
  const int32_t rot_dim_i = static_cast<int32_t>(rotary_dim);
  const int32_t head_size_i = static_cast<int32_t>(head_size);
  dim3 block(std::min<int64_t>(
      static_cast<int64_t>(num_heads) * (rot_dim_i / 2), 512));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  CHECK(positions.scalar_type() == torch::kInt32 ||
        positions.scalar_type() == torch::kInt64)
      << "partial_rotary_embedding_inplace: positions must be int32 or "
         "int64, got "
      << positions.scalar_type();
  const bool positions_is_int64 = positions.scalar_type() == torch::kInt64;

  DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "partial_rotary_embedding_inplace", [&] {
        scalar_t* query_ptr = query.data_ptr<scalar_t>();
        scalar_t* key_ptr = key.data_ptr<scalar_t>();
        const scalar_t* cache_ptr = cos_sin_cache.data_ptr<scalar_t>();
        if (positions_is_int64) {
          const int64_t* pos_ptr = positions.data_ptr<int64_t>();
          if (is_neox) {
            rotary_embedding_kernel<scalar_t, int64_t, true>
                <<<grid, block, 0, stream>>>(pos_ptr,
                                             query_ptr,
                                             key_ptr,
                                             cache_ptr,
                                             rot_dim_i,
                                             query_stride,
                                             key_stride,
                                             head_stride,
                                             num_heads,
                                             num_kv_heads,
                                             head_size_i);
          } else {
            rotary_embedding_kernel<scalar_t, int64_t, false>
                <<<grid, block, 0, stream>>>(pos_ptr,
                                             query_ptr,
                                             key_ptr,
                                             cache_ptr,
                                             rot_dim_i,
                                             query_stride,
                                             key_stride,
                                             head_stride,
                                             num_heads,
                                             num_kv_heads,
                                             head_size_i);
          }
        } else {
          const int32_t* pos_ptr = positions.data_ptr<int32_t>();
          if (is_neox) {
            rotary_embedding_kernel<scalar_t, int32_t, true>
                <<<grid, block, 0, stream>>>(pos_ptr,
                                             query_ptr,
                                             key_ptr,
                                             cache_ptr,
                                             rot_dim_i,
                                             query_stride,
                                             key_stride,
                                             head_stride,
                                             num_heads,
                                             num_kv_heads,
                                             head_size_i);
          } else {
            rotary_embedding_kernel<scalar_t, int32_t, false>
                <<<grid, block, 0, stream>>>(pos_ptr,
                                             query_ptr,
                                             key_ptr,
                                             cache_ptr,
                                             rot_dim_i,
                                             query_stride,
                                             key_stride,
                                             head_stride,
                                             num_heads,
                                             num_kv_heads,
                                             head_size_i);
          }
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xllm::kernel::cuda
