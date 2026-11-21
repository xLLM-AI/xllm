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

#include <c10/util/ScopeExit.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>

#include <array>
#include <cmath>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {
void validate_dual_affine_layer_norm(const torch::Tensor& x,
                                     const torch::Tensor& gamma1,
                                     const torch::Tensor& beta1,
                                     const torch::Tensor& gamma2,
                                     const torch::Tensor& beta2,
                                     double epsilon) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
              "DualAffineLayerNorm: inputs must be NPU tensors");
  TORCH_CHECK(x.dim() == 2, "DualAffineLayerNorm: x must have shape [N,H]");
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 4096 && x.size(1) >= 1 &&
                  x.size(1) <= 8192,
              "DualAffineLayerNorm: N or H outside supported range");
  TORCH_CHECK(
      x.scalar_type() == torch::kHalf || x.scalar_type() == torch::kBFloat16,
      "DualAffineLayerNorm: expected FP16 or BF16");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon >= 1e-7 && epsilon <= 1e-2,
              "DualAffineLayerNorm: epsilon must be in [1e-7,1e-2]");
  TORCH_CHECK(x.is_contiguous(), "DualAffineLayerNorm: x must be contiguous");
  for (const torch::Tensor* p : {&gamma1, &beta1, &gamma2, &beta2}) {
    TORCH_CHECK(p->device() == x.device() &&
                    p->scalar_type() == x.scalar_type() && p->dim() == 1 &&
                    p->size(0) == x.size(1) && p->is_contiguous(),
                "DualAffineLayerNorm: affine inputs must be contiguous [H] "
                "with the same device and dtype as x");
  }
}

std::tuple<torch::Tensor, torch::Tensor> dual_affine_layer_norm(
    const torch::Tensor& x,
    const torch::Tensor& gamma1,
    const torch::Tensor& beta1,
    const torch::Tensor& gamma2,
    const torch::Tensor& beta2,
    double epsilon) {
  validate_dual_affine_layer_norm(x, gamma1, beta1, gamma2, beta2, epsilon);
  const c10_npu::OptionalNPUGuard guard(x.device());
  const c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
  for (const torch::Tensor* input : {&x, &gamma1, &beta1, &gamma2, &beta2}) {
    c10_npu::NPUCachingAllocator::recordStream(input->storage().data_ptr(),
                                               stream);
  }
  torch::Tensor y1 =
      at_npu::native::OpPreparation::apply_tensor_without_format(x);
  torch::Tensor y2 =
      at_npu::native::OpPreparation::apply_tensor_without_format(x);
  static const auto create_tensor =
      aclnn::detail::get_op_api_func<aclnn::detail::AclCreateTensorFn>(
          "aclCreateTensor");
  static const auto destroy_tensor =
      aclnn::detail::get_op_api_func<aclnn::detail::AclDestroyTensorFn>(
          "aclDestroyTensor");
  using Prepare = int (*)(const aclTensor*,
                          const aclTensor*,
                          const aclTensor*,
                          const aclTensor*,
                          const aclTensor*,
                          double,
                          aclTensor*,
                          aclTensor*,
                          uint64_t*,
                          aclOpExecutor**);
  using Execute = int (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
  static const auto prepare = aclnn::detail::get_op_api_func<Prepare>(
      "aclnnDualAffineLayerNormGetWorkspaceSize");
  static const auto execute =
      aclnn::detail::get_op_api_func<Execute>("aclnnDualAffineLayerNorm");
  TORCH_CHECK(create_tensor && destroy_tensor && prepare && execute,
              "DualAffineLayerNorm ACLNN library is not installed");
  const aclrtStream acl_stream = stream.stream(false);
  static const auto init_memory =
      aclnn::detail::get_op_api_func<aclnn::detail::InitHugeMemThreadLocalFn>(
          "InitHugeMemThreadLocal");
  static const auto release_memory =
      aclnn::detail::get_op_api_func<aclnn::detail::ReleaseHugeMemFn>(
          "ReleaseHugeMem");
  static const auto uninit_memory =
      aclnn::detail::get_op_api_func<aclnn::detail::UnInitHugeMemThreadLocalFn>(
          "UnInitHugeMemThreadLocal");
  TORCH_CHECK(init_memory && release_memory && uninit_memory,
              "CANN temporary memory manager is not available");
  const aclDataType dtype =
      x.scalar_type() == torch::kHalf ? ACL_FLOAT16 : ACL_BF16;
  const std::array<void*, 7> pointers{x.data_ptr(),
                                      gamma1.data_ptr(),
                                      beta1.data_ptr(),
                                      gamma2.data_ptr(),
                                      beta2.data_ptr(),
                                      y1.data_ptr(),
                                      y2.data_ptr()};
  const int64_t rows = x.size(0), cols = x.size(1);
  // Copy metadata, as torch_npu's task queue does. The caching allocator tracks
  // stream order; retaining output Tensor objects here delays their reuse.
  auto launch = [pointers, rows, cols, dtype, epsilon, acl_stream]() -> int {
    init_memory(nullptr, false);
    const int64_t matrix_shape[] = {rows, cols}, matrix_stride[] = {cols, 1};
    const int64_t vector_shape[] = {cols}, vector_stride[] = {1};
    std::array<aclTensor*, 7> descriptors{};
    const auto cleanup = c10::make_scope_exit([&]() {
      for (const auto* descriptor : descriptors) {
        if (descriptor != nullptr) {
          destroy_tensor(descriptor);
        }
      }
      release_memory(nullptr, false);
      uninit_memory(nullptr, false);
    });
    for (size_t i = 0; i < pointers.size(); ++i) {
      const bool matrix = i == 0 || i >= 5;
      descriptors[i] = create_tensor(matrix ? matrix_shape : vector_shape,
                                     matrix ? 2 : 1,
                                     dtype,
                                     matrix ? matrix_stride : vector_stride,
                                     0,
                                     ACL_FORMAT_ND,
                                     matrix ? matrix_shape : vector_shape,
                                     matrix ? 2 : 1,
                                     pointers[i]);
      TORCH_CHECK(descriptors[i] != nullptr,
                  "DualAffineLayerNorm tensor descriptor failed");
    }
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    const int32_t prepare_status = prepare(descriptors[0],
                                           descriptors[1],
                                           descriptors[2],
                                           descriptors[3],
                                           descriptors[4],
                                           epsilon,
                                           descriptors[5],
                                           descriptors[6],
                                           &workspace_size,
                                           &executor);
    int32_t execute_status = 0;
    if (prepare_status == 0 && workspace_size == 0) {
      execute_status = execute(nullptr, 0, executor, acl_stream);
    }
    TORCH_CHECK(prepare_status == 0,
                "DualAffineLayerNorm prepare failed: ",
                prepare_status,
                " ",
                aclGetRecentErrMsg());
    TORCH_CHECK(workspace_size == 0,
                "DualAffineLayerNorm requires zero workspace");
    TORCH_CHECK(execute_status == 0,
                "DualAffineLayerNorm launch failed: ",
                execute_status,
                " ",
                aclGetRecentErrMsg());
    return execute_status;
  };
  at_npu::native::OpCommand::RunOpApiV2("aclnnDualAffineLayerNorm", launch);
  return {y1, y2};
}
}  // namespace xllm::kernel::npu
