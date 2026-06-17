/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <c10/cuda/CUDAStream.h>

#include "cuda_ops_api.h"
#include "device_utils.cuh"

namespace xllm::kernel::cuda {

template <typename T>
__global__ void XLLM_KERNEL_ATTR(1024) reshape_paged_cache_kernel(
    const int* __restrict__ slot_ids,  // [n_tokens]
    const T* __restrict__ keys,        // [n_tokens, n_heads, head_dim]
    const T* __restrict__ values,      // [n_tokens, n_heads, head_dim]
    T* __restrict__ key_cache,
    T* __restrict__ value_cache,
    int64_t k_stride,
    int64_t v_stride,
    int64_t n_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
  // block/token index
  const int64_t bid = blockIdx.x;
  // which slot to write to
  const int64_t slot_id = slot_ids[bid];
  if (slot_id < 0) {
    return;
  }
  // block index
  const int64_t block_idx = slot_id / block_size;
  // offset within block
  const int64_t block_offset = slot_id % block_size;
  // base index for the block in cache
  const int64_t block_base_idx = block_idx * block_size * n_kv_heads * head_dim;
  // copy value one by one for the token
  for (int64_t i = threadIdx.x; i < n_kv_heads * head_dim; i += blockDim.x) {
    const int64_t k_src_idx = bid * k_stride + i;
    const int64_t v_src_idx = bid * v_stride + i;
    // cache: [n_blocks, block_size, n_heads, head_dim]
    const int64_t head_base_idx =
        block_base_idx + block_offset * n_kv_heads * head_dim;
    // which head to write to
    const int head_idx = i / head_dim;
    // which dim within head to write to
    const int head_offset = i % head_dim;
    const int64_t dst_idx = head_base_idx + head_idx * head_dim + head_offset;
    key_cache[dst_idx] = keys[k_src_idx];
    value_cache[dst_idx] = values[v_src_idx];
  }
}

void reshape_paged_cache(
    torch::Tensor slot_ids,   // [n_tokens]
    torch::Tensor keys,       // [n_tokens, n_kv_heads, head_dim]
    torch::Tensor values,     // [n_tokens, n_kv_heads, head_dim]
    torch::Tensor key_cache,  // [n_blocks, block_size, n_heads, head_dim]
    torch::Tensor value_cache) {
  // keys and values should be continuous at n_kv_heads and head_dim dims
  CHECK(keys.stride(-1) == 1 && keys.stride(-2) == keys.size(-1));
  CHECK(values.stride(-1) == 1 && values.stride(-2) == values.size(-1));
  const int64_t n_tokens = keys.size(-3);
  const int64_t n_kv_heads = keys.size(-2);
  const int64_t head_dim = keys.size(-1);
  const int64_t block_size = key_cache.size(-3);
  // it is possible that keys and values have different strides
  const int64_t k_stride = keys.stride(-3);
  const int64_t v_stride = values.stride(-3);
  const int64_t n = n_kv_heads * head_dim;
  dim3 grid(n_tokens);
  dim3 block(std::min<int>(n, 1024));
  DISPATCH_FLOATING_TYPES(
      keys.scalar_type(), "reshape_paged_cache_kernel", [&] {
        reshape_paged_cache_kernel<scalar_t>
            <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                slot_ids.data_ptr<int>(),
                keys.data_ptr<scalar_t>(),
                values.data_ptr<scalar_t>(),
                key_cache.data_ptr<scalar_t>(),
                value_cache.data_ptr<scalar_t>(),
                k_stride,
                v_stride,
                n_kv_heads,
                head_dim,
                block_size);
      });
}

}  // namespace xllm::kernel::cuda
