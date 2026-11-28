// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include <c10/util/Logging.h>
#include <torch/library.h>
#include <torch/types.h>

#include <cstdint>

#include "op_plugin/utils/op_api_common_base.h"

namespace xllm_kernel {
namespace {

bool has_grouped_matmul_out() {
  return GetOpApiFuncAddr("aclnnGroupedMatmulV4") &&
         GetOpApiFuncAddr("aclnnGroupedMatmulV4GetWorkspaceSize");
}

torch::Tensor grouped_matmul_out(const torch::Tensor& x,
                                 const torch::Tensor& weight,
                                 const torch::Tensor& scale,
                                 const torch::Tensor& per_token_scale,
                                 const torch::Tensor& group_list,
                                 int64_t split_item,
                                 int64_t group_type,
                                 int64_t group_list_type,
                                 torch::Tensor& output) {
  CHECK(x.dim() == 2);
  CHECK(weight.dim() == 3);
  CHECK(x.size(1) == weight.size(1));
  CHECK(output.is_contiguous());
  CHECK(output.sizes().equals({x.size(0), weight.size(2)}));
  CHECK(output.device() == x.device());

  const torch::DeviceGuard guard(x.device());
  const torch::TensorList x_list(x);
  const torch::TensorList weight_list(weight);
  const torch::TensorList scale_list(scale);
  const torch::TensorList per_token_scale_list(per_token_scale);
  const torch::TensorList output_list(output);
  constexpr int64_t kNoActivation = 0;
  aclTensorList* absent = nullptr;
  EXEC_NPU_CMD_EXT(aclnnGroupedMatmulV4,
                   x_list,
                   weight_list,
                   /*biasOptional=*/absent,
                   scale_list,
                   /*offsetOptional=*/absent,
                   /*antiquantScaleOptional=*/absent,
                   /*antiquantOffsetOptional=*/absent,
                   per_token_scale_list,
                   group_list,
                   /*activationInputOptional=*/absent,
                   /*activationQuantScaleOptional=*/absent,
                   /*activationQuantOffsetOptional=*/absent,
                   split_item,
                   group_type,
                   group_list_type,
                   kNoActivation,
                   output_list,
                   /*activationFeatureOutOptional=*/absent,
                   /*dynQuantScaleOutOptional=*/absent);
  return output;
}

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("has_grouped_matmul_out() -> bool", TORCH_FN(has_grouped_matmul_out));
  m.def(
      "grouped_matmul_out(Tensor x, Tensor weight, Tensor scale, "
      "Tensor per_token_scale, Tensor group_list, int split_item, "
      "int group_type, int group_list_type, Tensor(a!) output) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("grouped_matmul_out", TORCH_FN(grouped_matmul_out));
}

}  // namespace
}  // namespace xllm_kernel
