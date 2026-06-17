/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <optional>
#include <string>
#include <tuple>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"

namespace xllm::kernel::npu {

bool has_dispatch_ffn_combine() {
  static const bool is_available =
      aclnn::detail::get_op_api_func_addr(
          "aclnnDispatchFFNCombineGetWorkspaceSize") != nullptr &&
      aclnn::detail::get_op_api_func_addr("aclnnDispatchFFNCombine") != nullptr;
  return is_available;
}

std::tuple<torch::Tensor, torch::Tensor> apply_npu_dispatch_ffn_combine(
    const torch::Tensor& x,
    const torch::TensorList weight1,
    const torch::TensorList weight2,
    const torch::Tensor& expert_ids,
    const torch::TensorList scale1,
    const torch::TensorList scale2,
    const torch::Tensor& probs,
    const std::string& group,
    int64_t max_output_size,
    double swiglu_limit,
    const std::optional<torch::Tensor>& output,
    const std::optional<torch::Tensor>& expert_token_nums) {
  TORCH_CHECK(has_dispatch_ffn_combine(),
              "aclnnDispatchFFNCombine is not available in libopapi.");
  TORCH_CHECK(x.dim() == 2, "DispatchFFNCombine expects 2D x.");
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "DispatchFFNCombine expects fp16/bf16 x, got ",
              c10::toString(x.scalar_type()));
  TORCH_CHECK(!weight1.empty(),
              "DispatchFFNCombine expects non-empty weight1 TensorList.");
  TORCH_CHECK(!weight2.empty(),
              "DispatchFFNCombine expects non-empty weight2 TensorList.");
  TORCH_CHECK(weight1.size() == weight2.size(),
              "DispatchFFNCombine weight1/weight2 list size mismatch: ",
              weight1.size(),
              " vs ",
              weight2.size());
  TORCH_CHECK(!scale1.empty(),
              "DispatchFFNCombine expects non-empty scale1 TensorList.");
  TORCH_CHECK(!scale2.empty(),
              "DispatchFFNCombine expects non-empty scale2 TensorList.");
  TORCH_CHECK(scale1.size() == weight1.size(),
              "DispatchFFNCombine scale1/weight1 list size mismatch: ",
              scale1.size(),
              " vs ",
              weight1.size());
  TORCH_CHECK(scale2.size() == weight2.size(),
              "DispatchFFNCombine scale2/weight2 list size mismatch: ",
              scale2.size(),
              " vs ",
              weight2.size());
  TORCH_CHECK(expert_ids.dim() == 2,
              "DispatchFFNCombine expects 2D expert_ids.");
  TORCH_CHECK(expert_ids.scalar_type() == at::kInt,
              "DispatchFFNCombine expects int32 expert_ids, got ",
              c10::toString(expert_ids.scalar_type()));
  TORCH_CHECK(probs.dim() == 2, "DispatchFFNCombine expects 2D probs.");
  TORCH_CHECK(probs.scalar_type() == at::kFloat,
              "DispatchFFNCombine expects float32 probs, got ",
              c10::toString(probs.scalar_type()));
  TORCH_CHECK(probs.sizes() == expert_ids.sizes(),
              "DispatchFFNCombine probs/expert_ids shape mismatch: ",
              probs.sizes(),
              " vs ",
              expert_ids.sizes());
  TORCH_CHECK(!group.empty(),
              "DispatchFFNCombine requires non-empty HCCL group name.");
  TORCH_CHECK(max_output_size > 0,
              "DispatchFFNCombine requires max_output_size > 0.");

  auto out = output.has_value() && output->defined() ? output.value()
                                                     : at::empty_like(x);
  TORCH_CHECK(out.sizes() == x.sizes(),
              "DispatchFFNCombine output shape mismatch: ",
              out.sizes(),
              " vs ",
              x.sizes());

  int64_t expert_token_nums_size = static_cast<int64_t>(weight1.size());
  auto expert_token_nums_out =
      expert_token_nums.has_value() && expert_token_nums->defined()
          ? expert_token_nums.value()
          : at::empty({expert_token_nums_size}, x.options().dtype(at::kInt));
  TORCH_CHECK(expert_token_nums_out.dim() == 1,
              "DispatchFFNCombine expects 1D expert_token_nums output.");
  TORCH_CHECK(expert_token_nums_out.scalar_type() == at::kInt,
              "DispatchFFNCombine expects int32 expert_token_nums output, got ",
              c10::toString(expert_token_nums_out.scalar_type()));

  std::string group_copy = group;
  char* group_ptr = group_copy.data();

  EXEC_NPU_CMD(aclnnDispatchFFNCombine,
               x,
               weight1,
               weight2,
               expert_ids,
               scale1,
               scale2,
               probs,
               group_ptr,
               max_output_size,
               swiglu_limit,
               out,
               expert_token_nums_out);

  return std::make_tuple(out, expert_token_nums_out);
}

}  // namespace xllm::kernel::npu
