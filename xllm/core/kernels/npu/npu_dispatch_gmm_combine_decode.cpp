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
namespace {

int64_t resolve_local_expert_num(int64_t ep_rank_size,
                                 int64_t ep_rank_id,
                                 int64_t moe_expert_num,
                                 int64_t shared_expert_rank_num) {
  const bool is_shared_expert_rank = ep_rank_id < shared_expert_rank_num;
  if (is_shared_expert_rank) {
    return 1;
  }
  const int64_t routed_ep_size = ep_rank_size - shared_expert_rank_num;
  TORCH_CHECK(routed_ep_size > 0,
              "invalid shared_expert_rank_num=",
              shared_expert_rank_num,
              " for ep_rank_size=",
              ep_rank_size);
  TORCH_CHECK(moe_expert_num % routed_ep_size == 0,
              "moe_expert_num must be divisible by routed EP size, got ",
              moe_expert_num,
              " / ",
              routed_ep_size);
  return moe_expert_num / routed_ep_size;
}

}  // namespace

bool has_dispatch_gmm_combine_decode() {
  static const bool is_available =
      aclnn::detail::get_op_api_func_addr(
          "aclnnDispatchGmmCombineDecodeGetWorkspaceSize") != nullptr &&
      aclnn::detail::get_op_api_func_addr("aclnnDispatchGmmCombineDecode") !=
          nullptr;
  return is_available;
}

std::tuple<torch::Tensor, torch::Tensor> apply_npu_dispatch_gmm_combine_decode(
    const torch::Tensor& x,
    const torch::Tensor& expert_ids,
    const torch::TensorList gmm1_permuted_weight,
    const torch::TensorList gmm1_permuted_weight_scale,
    const torch::TensorList gmm2_weight,
    const torch::TensorList gmm2_weight_scale,
    const torch::Tensor& expert_scales,
    const std::optional<torch::Tensor>& expert_smooth_scales,
    const std::optional<torch::Tensor>& x_active_mask,
    const std::string& group_ep,
    int64_t ep_rank_size,
    int64_t ep_rank_id,
    int64_t moe_expert_num,
    int64_t shared_expert_num,
    int64_t shared_expert_rank_num,
    int64_t quant_mode,
    int64_t global_bs) {
  TORCH_CHECK(has_dispatch_gmm_combine_decode(),
              "aclnnDispatchGmmCombineDecode is not available in libopapi.");
  TORCH_CHECK(x.dim() == 2, "DispatchGmmCombineDecode expects 2D x.");
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "DispatchGmmCombineDecode expects fp16/bf16 x, got ",
              c10::toString(x.scalar_type()));
  TORCH_CHECK(expert_ids.dim() == 2,
              "DispatchGmmCombineDecode expects 2D expert_ids.");
  TORCH_CHECK(expert_ids.scalar_type() == at::kInt,
              "DispatchGmmCombineDecode expects int32 expert_ids, got ",
              c10::toString(expert_ids.scalar_type()));
  TORCH_CHECK(expert_scales.dim() == 2,
              "DispatchGmmCombineDecode expects 2D expert_scales.");
  TORCH_CHECK(expert_scales.scalar_type() == at::kFloat,
              "DispatchGmmCombineDecode expects float32 expert_scales, got ",
              c10::toString(expert_scales.scalar_type()));
  TORCH_CHECK(expert_scales.sizes() == expert_ids.sizes(),
              "DispatchGmmCombineDecode expert_scales/expert_ids shape "
              "mismatch: ",
              expert_scales.sizes(),
              " vs ",
              expert_ids.sizes());
  TORCH_CHECK(!gmm1_permuted_weight.empty(),
              "DispatchGmmCombineDecode expects non-empty gmm1 weight list.");
  TORCH_CHECK(!gmm2_weight.empty(),
              "DispatchGmmCombineDecode expects non-empty gmm2 weight list.");
  TORCH_CHECK(gmm1_permuted_weight.size() == gmm2_weight.size(),
              "DispatchGmmCombineDecode gmm1/gmm2 list size mismatch: ",
              gmm1_permuted_weight.size(),
              " vs ",
              gmm2_weight.size());
  TORCH_CHECK(gmm1_permuted_weight_scale.size() == gmm1_permuted_weight.size(),
              "DispatchGmmCombineDecode gmm1 scale/weight list size "
              "mismatch: ",
              gmm1_permuted_weight_scale.size(),
              " vs ",
              gmm1_permuted_weight.size());
  TORCH_CHECK(gmm2_weight_scale.size() == gmm2_weight.size(),
              "DispatchGmmCombineDecode gmm2 scale/weight list size "
              "mismatch: ",
              gmm2_weight_scale.size(),
              " vs ",
              gmm2_weight.size());
  TORCH_CHECK(!group_ep.empty(),
              "DispatchGmmCombineDecode requires non-empty EP group name.");
  TORCH_CHECK(ep_rank_size > 1,
              "DispatchGmmCombineDecode requires ep_rank_size > 1.");
  TORCH_CHECK(ep_rank_id >= 0 && ep_rank_id < ep_rank_size,
              "invalid EP rank ",
              ep_rank_id,
              " for ep_rank_size ",
              ep_rank_size);

  torch::Tensor output = at::empty_like(x);
  const int64_t local_expert_num = resolve_local_expert_num(
      ep_rank_size, ep_rank_id, moe_expert_num, shared_expert_rank_num);
  torch::Tensor expert_token_nums =
      at::empty({local_expert_num}, expert_ids.options().dtype(at::kLong));

  auto expert_smooth_scales_optional =
      expert_smooth_scales.has_value() && expert_smooth_scales->defined()
          ? c10::optional<at::Tensor>(expert_smooth_scales.value())
          : c10::nullopt;
  auto x_active_mask_optional =
      x_active_mask.has_value() && x_active_mask->defined()
          ? c10::optional<at::Tensor>(x_active_mask.value())
          : c10::nullopt;

  std::string group_ep_copy = group_ep;
  char* group_ep_ptr = group_ep_copy.data();

  EXEC_NPU_CMD(aclnnDispatchGmmCombineDecode,
               x,
               expert_ids,
               gmm1_permuted_weight,
               gmm1_permuted_weight_scale,
               gmm2_weight,
               gmm2_weight_scale,
               expert_scales,
               expert_smooth_scales_optional,
               x_active_mask_optional,
               group_ep_ptr,
               ep_rank_size,
               ep_rank_id,
               moe_expert_num,
               shared_expert_num,
               shared_expert_rank_num,
               quant_mode,
               global_bs,
               output,
               expert_token_nums);

  return std::make_tuple(output, expert_token_nums);
}

}  // namespace xllm::kernel::npu
