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

#include <acl/acl.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>

#include <algorithm>
#include <array>

#include "aclrtlaunch_dual_affine_layer_norm_bf16.h"
#include "aclrtlaunch_dual_affine_layer_norm_fp16.h"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {
std::tuple<torch::Tensor, torch::Tensor> dual_affine_layer_norm_direct(
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
  static const int64_t core_count = [&]() {
    int64_t count = 0;
    TORCH_CHECK(aclrtGetDeviceInfo(x.get_device(),
                                   ACL_DEV_ATTR_VECTOR_CORE_NUM,
                                   &count) == ACL_SUCCESS,
                "query Vector Core count failed");
    return count;
  }();
  const uint32_t rows = x.size(0), cols = x.size(1);
  const uint32_t cores = std::min<int64_t>(rows, core_count);
  torch::Tensor y1 =
      at_npu::native::OpPreparation::apply_tensor_without_format(x);
  torch::Tensor y2 =
      at_npu::native::OpPreparation::apply_tensor_without_format(x);
  const aclrtStream acl_stream = stream.stream(false);
  const std::array<void*, 7> pointers{x.data_ptr(),
                                      gamma1.data_ptr(),
                                      beta1.data_ptr(),
                                      gamma2.data_ptr(),
                                      beta2.data_ptr(),
                                      y1.data_ptr(),
                                      y2.data_ptr()};
  const auto function = x.scalar_type() == torch::kHalf
                            ? aclrtlaunch_dual_affine_layer_norm_fp16
                            : aclrtlaunch_dual_affine_layer_norm_bf16;
  auto launch =
      [pointers, function, cores, rows, cols, epsilon, acl_stream]() -> int {
    const uint32_t status = function(cores,
                                     acl_stream,
                                     pointers[0],
                                     pointers[1],
                                     pointers[2],
                                     pointers[3],
                                     pointers[4],
                                     pointers[5],
                                     pointers[6],
                                     rows,
                                     cols,
                                     cores,
                                     static_cast<float>(epsilon));
    TORCH_CHECK(status == ACL_SUCCESS, "direct kernel launch failed: ", status);
    return status;
  };
  at_npu::native::OpCommand::RunOpApiV2("DualAffineLayerNormDirect", launch);
  return {y1, y2};
}
}  // namespace xllm::kernel::npu
TORCH_LIBRARY_FRAGMENT(xllm_ops, m) {
  m.def(
      "dual_affine_layer_norm(Tensor x, Tensor gamma1, Tensor beta1, Tensor "
      "gamma2, Tensor beta2, float epsilon=1e-5) -> (Tensor, Tensor)");
  m.def(
      "dual_affine_layer_norm_direct(Tensor x, Tensor gamma1, Tensor beta1, "
      "Tensor gamma2, Tensor beta2, float epsilon=1e-5) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(xllm_ops, PrivateUse1, m) {
  m.impl("dual_affine_layer_norm", &xllm::kernel::npu::dual_affine_layer_norm);
  m.impl("dual_affine_layer_norm_direct",
         &xllm::kernel::npu::dual_affine_layer_norm_direct);
}
