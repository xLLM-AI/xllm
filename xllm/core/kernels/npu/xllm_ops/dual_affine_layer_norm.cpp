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

// DualAffineLayerNorm: both branches share one FP32 row mean/centered
// variance and then apply their own affine parameters. The operator ships as a
// custom ACLNN package (vendor `dual_affine_layernorm_custom`); EXEC_NPU_CMD
// resolves `aclnnDualAffineLayerNormGetWorkspaceSize`/
// `aclnnDualAffineLayerNorm` from the installed `libcust_opapi.so`, so no extra
// link-time dependency is added here.

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

bool is_dual_affine_dtype(torch::ScalarType dtype) {
  return dtype == torch::kHalf || dtype == torch::kBFloat16;
}

void check_dual_affine_layer_norm_inputs(const torch::Tensor& x,
                                         const torch::Tensor& gamma1,
                                         const torch::Tensor& beta1,
                                         const torch::Tensor& gamma2,
                                         const torch::Tensor& beta2,
                                         double eps) {
  CHECK(x.defined() && gamma1.defined() && beta1.defined() &&
        gamma2.defined() && beta2.defined())
      << "dual_affine_layer_norm: all inputs must be defined";
  CHECK(x.device().is_privateuseone())
      << "dual_affine_layer_norm: x must be an NPU tensor";
  CHECK(x.dim() == 2) << "dual_affine_layer_norm: x must be [N, H]";
  CHECK(is_dual_affine_dtype(x.scalar_type()))
      << "dual_affine_layer_norm: x dtype must be float16 or bfloat16, got "
      << x.scalar_type();
  // The ACLNN converter preserves FRACTAL_NZ, so reject packed storage here
  // instead of forwarding a descriptor the kernel cannot consume.
  CHECK(at_npu::native::custom_ops::get_npu_format(x) == ACL_FORMAT_ND)
      << "dual_affine_layer_norm: x must use ND storage format";
  CHECK(x.is_contiguous()) << "dual_affine_layer_norm: x must be contiguous";
  const int64_t n = x.size(0);
  const int64_t h = x.size(1);
  CHECK(n >= 1 && n <= 4096)
      << "dual_affine_layer_norm: N must be in [1, 4096], got " << n;
  CHECK(h >= 1 && h <= 8192)
      << "dual_affine_layer_norm: H must be in [1, 8192], got " << h;
  for (const torch::Tensor* param : {&gamma1, &beta1, &gamma2, &beta2}) {
    CHECK(param->dim() == 1 && param->size(0) == h)
        << "dual_affine_layer_norm: affine parameters must be [H]";
    CHECK(param->scalar_type() == x.scalar_type())
        << "dual_affine_layer_norm: affine parameters must share x dtype";
    CHECK(param->device() == x.device())
        << "dual_affine_layer_norm: all tensors must be on the same device";
    CHECK(param->is_contiguous())
        << "dual_affine_layer_norm: affine parameters must be contiguous";
  }
  CHECK(eps >= 1e-7 && eps <= 1e-2)
      << "dual_affine_layer_norm: epsilon must be in [1e-7, 1e-2], got " << eps;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> dual_affine_layer_norm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& gamma1,
    const torch::Tensor& beta1,
    const torch::Tensor& gamma2,
    const torch::Tensor& beta2,
    double eps) {
  check_dual_affine_layer_norm_inputs(x, gamma1, beta1, gamma2, beta2, eps);

  // Two independent storage allocations are required by the operator contract.
  torch::Tensor y1 = torch::empty_like(x);
  torch::Tensor y2 = torch::empty_like(x);
  const float eps_f = static_cast<float>(eps);

  EXEC_NPU_CMD(
      aclnnDualAffineLayerNorm, x, gamma1, beta1, gamma2, beta2, eps_f, y1, y2);

  return {y1, y2};
}

}  // namespace xllm::kernel::npu
