/* Copyright 2025-2026 The xLLM Authors.

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

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>

#include "acl/acl.h"
#include "dispatch_registry.h"
#include "tilelang_ops_api.h"

#ifndef XLLM_TL_EMBEDDING_REGISTRY_INC
#error "XLLM_TL_EMBEDDING_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_EMBEDDING_REGISTRY_INC

EmbeddingSpecialization build_runtime_specialization(
    const torch::Tensor& weight) {
  return make_embedding_specialization(
      EmbeddingVocabSize{static_cast<int32_t>(weight.size(0))},
      EmbeddingHiddenDim{static_cast<int32_t>(weight.size(1))},
      EmbeddingDType{to_tilelang_dtype(weight.scalar_type())});
}

}  // namespace

torch::Tensor embedding(const torch::Tensor& weight,
                        const torch::Tensor& token_ids) {
  CHECK(weight.defined() && token_ids.defined());
  CHECK(weight.device().type() == c10::DeviceType::PrivateUse1 &&
        token_ids.device().type() == c10::DeviceType::PrivateUse1);
  CHECK_EQ(weight.dim(), 2);
  CHECK_EQ(token_ids.dim(), 1);
  CHECK_EQ(weight.scalar_type(), c10::ScalarType::BFloat16);
  CHECK_EQ(token_ids.scalar_type(), c10::ScalarType::Int);
  CHECK(weight.is_contiguous());
  CHECK(token_ids.is_contiguous());
  CHECK_GT(token_ids.size(0), 0);
  CHECK_LE(token_ids.size(0), 192);

  const auto* entry =
      find_embedding_kernel_entry(build_runtime_specialization(weight));
  CHECK(entry != nullptr) << "TileLang embedding: no compiled variant";

  torch::Tensor output =
      torch::empty({token_ids.size(0), weight.size(1)}, weight.options());
  aclrtStream stream = c10_npu::getCurrentNPUStream(weight.device().index())
                           .stream(/*need_empty=*/true);
  entry->fn(reinterpret_cast<uint8_t*>(const_cast<void*>(weight.data_ptr())),
            reinterpret_cast<uint8_t*>(const_cast<void*>(token_ids.data_ptr())),
            reinterpret_cast<uint8_t*>(output.data_ptr()),
            static_cast<int32_t>(token_ids.size(0)),
            stream);
  return output;
}

}  // namespace xllm::kernel::npu::tilelang
