/* Copyright 2026 The xLLM Authors.

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

#include <limits>

#include "core/kv_cache/storage/kv_cache_create_options.h"
#include "core/kv_cache/storage/kv_cache_tensors.h"
#if defined(USE_NPU)
#include "acl/acl_base.h"
#endif

namespace xllm {

class KVCacheShape;

KVCacheTensors create_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options);

IndexedKVCacheTensors create_indexed_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options);

QuantizedKVCacheTensors create_quantized_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options);

LinearAttentionKVCacheTensors create_linear_attention_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options);

// Allocation and slot-reset both mark the KPool tail's gate plane (dim 1,
// index 1) with -inf so unwritten circular-tail rows read as invalid. Pool
// completion treats any finite gate as an initialized slot. The sentinel is
// an NPU-only layout contract; other platforms leave the tail zeroed.
inline void fill_kpool_tail_gate_sentinel(const torch::Tensor& kpool_tail) {
#if defined(USE_NPU)
  kpool_tail.select(/*dim=*/1, /*index=*/1)
      .fill_(-std::numeric_limits<float>::infinity());
#else
  (void)kpool_tail;
#endif
}

#if defined(USE_NPU)
aclFormat get_npu_kv_cache_format(const std::string& model_type);

// Allocate an NPU tensor from the huge-page device allocator. The returned
// tensor owns the ACL allocation and carries the requested NPU format.
torch::Tensor alloc_npu_huge_page_tensor(const std::vector<int64_t>& dims,
                                         torch::ScalarType dtype,
                                         aclFormat format);
#endif

}  // namespace xllm
