/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {

void reshape_and_cache_a5(const torch::Tensor& key,
                          const torch::Tensor& value,
                          torch::Tensor& key_cache,
                          torch::Tensor& value_cache,
                          const torch::Tensor& slot_mapping) {
  CHECK_EQ(key.dim(), 3) << "key must be [num_tokens, num_heads, head_dim].";
  CHECK_EQ(value.sizes(), key.sizes()) << "value shape must match key.";
  CHECK_EQ(key_cache.dim(), 4)
      << "key_cache must be [num_blocks, block_size, num_heads, head_dim].";
  CHECK_EQ(value_cache.sizes(), key_cache.sizes())
      << "value_cache shape must match key_cache.";
  CHECK_EQ(key.size(1), key_cache.size(2)) << "KV head count mismatch.";
  CHECK_EQ(key.size(2), key_cache.size(3)) << "KV head dimension mismatch.";
  CHECK_EQ(slot_mapping.dim(), 1) << "slot_mapping must be one-dimensional.";
  CHECK_EQ(slot_mapping.numel(), key.size(0))
      << "slot_mapping length must match num_tokens.";
  CHECK_EQ(slot_mapping.scalar_type(), torch::kInt32)
      << "slot_mapping must use int32.";
  CHECK(key.is_contiguous()) << "key must be contiguous.";
  CHECK(value.is_contiguous()) << "value must be contiguous.";
  CHECK(key_cache.is_contiguous()) << "key_cache must be contiguous.";
  CHECK(value_cache.is_contiguous()) << "value_cache must be contiguous.";
  CHECK(slot_mapping.is_contiguous()) << "slot_mapping must be contiguous.";
  CHECK(key.scalar_type() == torch::kFloat16 ||
        key.scalar_type() == torch::kBFloat16)
      << "A5 reshape_and_cache supports FP16 and BF16.";
  CHECK_EQ(key.scalar_type(), value.scalar_type());
  CHECK_EQ(key.scalar_type(), key_cache.scalar_type());
  CHECK_EQ(key.scalar_type(), value_cache.scalar_type());
  CHECK_EQ(key.device(), value.device());
  CHECK_EQ(key.device(), key_cache.device());
  CHECK_EQ(key.device(), value_cache.device());
  CHECK_EQ(key.device(), slot_mapping.device());
  CHECK(key.device().is_privateuseone())
      << "A5 reshape_and_cache inputs must be NPU tensors.";

  if (key.numel() == 0) {
    return;
  }
  EXEC_NPU_CMD(aclnnReshapeAndCacheA5,
               key,
               value,
               key_cache,
               value_cache,
               slot_mapping,
               key_cache,
               value_cache);
}

}  // namespace xllm::kernel::npu
