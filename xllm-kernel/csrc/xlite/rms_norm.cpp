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
#include <c10/core/DeviceGuard.h>
#include <c10/util/Logging.h>
#include <torch/library.h>
#include <torch/types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

// GVirt exports these generated host launchers but does not install their
// headers. Keep this signature aligned with NORM_FUNC_DEFINE in
// GVirt 4a6dc3102d5778928e4d21116bc4d4694d51e584,
// xlite/csrc/kernels/norm.h (NormKind::Rms == 0). Calling the launcher avoids
// XRuntime's separate stream, tensor pool, and device lifecycle ownership.
using XliteNormLauncher = uint32_t(uint32_t,
                                   aclrtStream,
                                   void*,
                                   void*,
                                   void*,
                                   void*,
                                   void*,
                                   uint32_t,
                                   uint32_t,
                                   float,
                                   int,
                                   uint32_t,
                                   uint32_t,
                                   uint32_t,
                                   uint32_t,
                                   uint32_t,
                                   bool,
                                   void*,
                                   uint32_t,
                                   bool);
extern "C" XliteNormLauncher aclrtlaunch_norm_bfloat16_t;
extern "C" XliteNormLauncher aclrtlaunch_norm_float16_t;

namespace {

void check_operands(const torch::Tensor& input,
                    const torch::Tensor& weight,
                    double eps) {
  CHECK((input.device().type()) == (c10::DeviceType::PrivateUse1));
  CHECK((input.device()) == (weight.device()));
  CHECK(input.scalar_type() == torch::kBFloat16 ||
        input.scalar_type() == torch::kHalf)
      << "xlite RMSNorm requires FP16/BF16 input";
  CHECK((input.scalar_type()) == (weight.scalar_type()));
  CHECK((input.dim()) >= (1));
  CHECK((weight.dim()) == (1));
  CHECK((input.size(-1)) == (weight.numel()));
  CHECK((weight.numel()) > (0));
  CHECK((weight.numel()) <= (8192))
      << "xlite affine RMSNorm exceeds UB capacity";
  CHECK((weight.numel() % 64) == (0))
      << "xlite RMSNorm requires a width divisible by 64";
  // GVirt computes row offsets in uint32_t, not just the row count.
  CHECK((input.numel()) <= (std::numeric_limits<uint32_t>::max()));
  CHECK(std::isfinite(eps) && eps >= 0.0 &&
        eps <= std::numeric_limits<float>::max());
  CHECK((at_npu::native::get_npu_format(input)) == (ACL_FORMAT_ND));
  CHECK((at_npu::native::get_npu_format(weight)) == (ACL_FORMAT_ND));
}

torch::Tensor xlite_rms_norm(const torch::Tensor& input,
                             const torch::Tensor& weight,
                             double eps) {
  check_operands(input, weight, eps);
  const c10::DeviceGuard device_guard(input.device());
  torch::Tensor contiguous_input = input.contiguous();
  torch::Tensor contiguous_weight = weight.contiguous();
  torch::Tensor output = torch::empty(input.sizes(), input.options());
  if (input.numel() == 0) {
    return output;
  }

  const int64_t rows = input.numel() / input.size(-1);
  const uint32_t width = static_cast<uint32_t>(input.size(-1));
  const uint32_t tokens = static_cast<uint32_t>(rows);
  int64_t vector_cores = 0;
  CHECK((aclGetDeviceCapability(input.device().index(),
                                ACL_DEVICE_INFO_VECTOR_CORE_NUM,
                                &vector_cores)) == (ACL_SUCCESS));
  CHECK((vector_cores) > (0));
  // GVirt distributes rows across blocks. Idle blocks add device overhead
  // without contributing work for decode-sized inputs.
  const uint32_t blocks = static_cast<uint32_t>(std::min(rows, vector_cores));
  const aclrtStream stream =
      c10_npu::getCurrentNPUStream(input.device().index()).stream();
  XliteNormLauncher* launcher = input.scalar_type() == torch::kBFloat16
                                    ? aclrtlaunch_norm_bfloat16_t
                                    : aclrtlaunch_norm_float16_t;
  auto launch = [contiguous_input,
                 contiguous_weight,
                 output,
                 tokens,
                 width,
                 eps,
                 blocks,
                 stream,
                 launcher]() -> int {
    const uint32_t status = launcher(blocks,
                                     stream,
                                     contiguous_input.data_ptr(),
                                     /*add_in_out=*/nullptr,
                                     contiguous_weight.data_ptr(),
                                     /*bias=*/nullptr,
                                     output.data_ptr(),
                                     tokens,
                                     width,
                                     static_cast<float>(eps),
                                     /*NormKind::Rms=*/0,
                                     /*cnt_per_token=*/1,
                                     /*in_step=*/width,
                                     /*out_step=*/width,
                                     /*in_start_offset=*/0,
                                     /*out_start_offset=*/0,
                                     /*use_norm=*/true,
                                     /*variance=*/nullptr,
                                     /*tp_size=*/1,
                                     /*out_fp32=*/false);
    CHECK((status) == (ACL_SUCCESS))
        << "xlite RMSNorm launch failed: " << aclGetRecentErrMsg();
    return static_cast<int>(status);
  };
  // The OpApi queue releases the callback after dispatch. The ACL compile
  // queue can retain callbacks in reusable slots, retaining captured tensors.
  // Both paths preserve ordering with preceding torch_npu operations.
  at_npu::native::OpCommand::RunOpApi("XliteRmsNorm", std::move(launch));
  return output;
}

}  // namespace

TORCH_LIBRARY(xllm_kernel, m) {
  m.def("xlite_rms_norm(Tensor input, Tensor weight, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("xlite_rms_norm", &xlite_rms_norm);
}
