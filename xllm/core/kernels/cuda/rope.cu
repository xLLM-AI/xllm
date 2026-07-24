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
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cuda_ops_api.h"
#include "device_utils.cuh"

// ref to:
// https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding_kernels.cu

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
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = *(cos_ptr + x_index);
    sin = *(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
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
inline __device__ void apply_rotary_embedding(
    scalar_t* __restrict__ query,  // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,    // nullptr or
                                   // [batch_size, seq_len, num_kv_heads,
                                   // head_size] or [num_tokens, num_kv_heads,
                                   // head_size]
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

template <typename scalar_t, bool IS_NEOX>
__global__ void XLLM_KERNEL_ATTR(512) rotary_embedding_kernel(
    const int64_t* __restrict__ positions,  // [batch_size, seq_len] or
                                            // [num_tokens]
    scalar_t* __restrict__ query,           // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,  // nullptr or
                                 // [batch_size, seq_len, num_kv_heads,
    // head_size] or [num_tokens, num_kv_heads,
    // head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2,
                                                 // rot_dim // 2]
    const int rot_dim,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t head_stride,
    const int num_heads,
    const int num_kv_heads,
    const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
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

// flashinfer rope ops
// void apply_rope_pos_ids_cos_sin_cache(torch::Tensor q,
//                                       torch::Tensor k,
//                                       torch::Tensor cos_sin_cache,
//                                       torch::Tensor pos_ids,
//                                       bool interleave) {
//   const int64_t head_dim = cos_sin_cache.size(-1) / 2;
//   q = q.view({q.size(0), -1, head_dim});
//   k = k.view({k.size(0), -1, head_dim});

//   FunctionFactory::get_instance().rope_func("rope").call(
//       q, k, q, k, cos_sin_cache, pos_ids, interleave);
// }

void rotary_embedding(
    torch::Tensor& positions,  // [batch_size, seq_len] or [num_tokens]
    torch::Tensor& query,  // [batch_size, seq_len, num_heads * head_size] or
                           // [num_tokens, num_heads * head_size] or
                           // [batch_size, seq_len, num_heads, head_size] or
                           // [num_tokens, num_heads, head_size]
    std::optional<torch::Tensor> key,
    // null or
    // [batch_size, seq_len, num_kv_heads * head_size] or
    // [num_tokens, num_kv_heads * head_size] or
    // [batch_size, seq_len, num_heads, head_size] or
    // [num_tokens, num_heads, head_size]
    // int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox) {
  // num_tokens = batch_size * seq_len
  const int positions_ndim = positions.dim();
  const int query_ndim = query.dim();
  // For partial rotary models, e.g. MiniMax-M2 with head_dim=128 and
  // rotary_dim=64, the cache width is the rotary dimension rather than the
  // physical per-head stride. When query is already shaped as
  // [*, num_heads, head_size], infer the real head_size from query itself.
  int64_t head_size = (query_ndim == positions_ndim + 2)
                          ? query.size(-1)
                          : cos_sin_cache.size(-1);
  int64_t num_tokens = positions.numel();

  // Make sure num_tokens dim is consistent across positions, query, and key
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

  // Make sure head_size is valid for query and key
  // hidden_size = num_heads * head_size
  int query_hidden_size = query.numel() / num_tokens;
  int key_hidden_size = key.has_value() ? key->numel() / num_tokens : 0;
  CHECK(query_hidden_size % head_size == 0);
  CHECK(key_hidden_size % head_size == 0);

  // Make sure query and key have consistent number of heads
  int num_heads = query_hidden_size / head_size;
  int num_kv_heads = key.has_value() ? key_hidden_size / head_size : num_heads;
  CHECK(num_heads % num_kv_heads == 0);

  int rot_dim = cos_sin_cache.size(1);
  int seq_dim_idx = positions_ndim - 1;
  int64_t query_stride = query.stride(seq_dim_idx);
  int64_t key_stride = key.has_value() ? key->stride(seq_dim_idx) : 0;
  // Determine head stride: for [*, heads, head_size] use stride of last dim;
  // for flat [*, heads*head_size], heads blocks are contiguous of size
  // head_size
  int64_t head_stride =
      (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;

  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "apply_rope_pos_ids_cos_sin_cache", [&] {
        if (is_neox) {
          rotary_embedding_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
              positions.data_ptr<int64_t>(),
              query.data_ptr<scalar_t>(),
              key.has_value() ? key->data_ptr<scalar_t>() : nullptr,
              cos_sin_cache.data_ptr<scalar_t>(),
              rot_dim,
              query_stride,
              key_stride,
              head_stride,
              num_heads,
              num_kv_heads,
              head_size);
        } else {
          rotary_embedding_kernel<scalar_t, false><<<grid, block, 0, stream>>>(
              positions.data_ptr<int64_t>(),
              query.data_ptr<scalar_t>(),
              key.has_value() ? key->data_ptr<scalar_t>() : nullptr,
              cos_sin_cache.data_ptr<scalar_t>(),
              rot_dim,
              query_stride,
              key_stride,
              head_stride,
              num_heads,
              num_kv_heads,
              head_size);
        }
      });
}

}  // namespace xllm::kernel::cuda
