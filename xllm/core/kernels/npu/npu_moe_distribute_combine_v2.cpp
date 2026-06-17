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

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"

namespace xllm::kernel::npu {
namespace {

constexpr int64_t kOutDtypeDefault = 0;
constexpr int64_t kGroupListTypeDefault = 0;

bool has_combine_v2() {
  static const bool is_available =
      aclnn::detail::get_op_api_func_addr(
          "aclnnMoeDistributeCombineV2GetWorkspaceSize") != nullptr &&
      aclnn::detail::get_op_api_func_addr("aclnnMoeDistributeCombineV2") !=
          nullptr;
  return is_available;
}

}  // namespace

torch::Tensor apply_npu_moe_distribute_combine_v2(
    const torch::Tensor& expand_x,
    const torch::Tensor& expert_ids,
    const torch::Tensor& assist_info_for_combine,
    const torch::Tensor& ep_send_counts,
    const torch::Tensor& expert_scales,
    const std::optional<torch::Tensor>& tp_send_counts,
    const std::optional<torch::Tensor>& x_active_mask,
    const std::optional<torch::Tensor>& expand_scales,
    const std::optional<torch::Tensor>& shared_expert_x,
    const std::string& group_ep,
    int64_t ep_world_size,
    int64_t ep_rank_id,
    int64_t moe_expert_num,
    const std::string& group_tp,
    int64_t tp_world_size,
    int64_t tp_rank_id,
    int64_t expert_shard_type,
    int64_t shared_expert_num,
    int64_t shared_expert_rank_num,
    int64_t global_bs,
    int64_t comm_quant_mode,
    const std::string& comm_alg) {
  CHECK(has_combine_v2())
      << "aclnnMoeDistributeCombineV2 is not available in libopapi.";
  CHECK_EQ(expand_x.dim(), 2) << "MoeDistributeCombineV2 expects 2D expand_x.";
  CHECK_EQ(expert_ids.dim(), 2)
      << "MoeDistributeCombineV2 expects 2D expert_ids.";
  CHECK_EQ(expert_ids.scalar_type(), torch::kInt32)
      << "MoeDistributeCombineV2 expects int32 expert_ids, got "
      << c10::toString(expert_ids.scalar_type());
  CHECK(expert_scales.defined())
      << "MoeDistributeCombineV2 requires expert_scales.";
  CHECK_EQ(expert_scales.scalar_type(), torch::kFloat32)
      << "MoeDistributeCombineV2 expects float32 expert_scales, got "
      << c10::toString(expert_scales.scalar_type());
  CHECK(!group_ep.empty())
      << "MoeDistributeCombineV2 requires non-empty EP group name.";
  CHECK_GT(ep_world_size, 1)
      << "MoeDistributeCombineV2 requires ep_world_size > 1.";
  CHECK(ep_rank_id >= 0 && ep_rank_id < ep_world_size)
      << "invalid EP rank " << ep_rank_id << " for ep_world_size "
      << ep_world_size;

  torch::Tensor output =
      torch::empty({expert_ids.size(0), expand_x.size(1)}, expand_x.options());

  auto tp_send_counts_optional =
      tp_world_size > 1 && tp_send_counts.has_value() &&
              tp_send_counts->defined()
          ? c10::optional<at::Tensor>(tp_send_counts.value())
          : c10::nullopt;
  auto x_active_mask_optional =
      x_active_mask.has_value() && x_active_mask->defined()
          ? c10::optional<at::Tensor>(x_active_mask.value())
          : c10::nullopt;
  auto expand_scales_optional =
      expand_scales.has_value() && expand_scales->defined()
          ? c10::optional<at::Tensor>(expand_scales.value())
          : c10::nullopt;
  auto shared_expert_x_optional =
      shared_expert_x.has_value() && shared_expert_x->defined()
          ? c10::optional<at::Tensor>(shared_expert_x.value())
          : c10::nullopt;
  c10::optional<at::Tensor> activation_scale_optional = c10::nullopt;
  c10::optional<at::Tensor> weight_scale_optional = c10::nullopt;
  c10::optional<at::Tensor> group_list_optional = c10::nullopt;

  std::string group_ep_copy = group_ep;
  char* group_ep_ptr = group_ep_copy.data();
  std::string group_tp_copy = group_tp;
  char* group_tp_ptr = group_tp_copy.data();
  std::string comm_alg_copy = comm_alg;
  char* comm_alg_ptr = comm_alg_copy.data();

  EXEC_NPU_CMD(aclnnMoeDistributeCombineV2,
               expand_x,
               expert_ids,
               assist_info_for_combine,
               ep_send_counts,
               expert_scales,
               tp_send_counts_optional,
               x_active_mask_optional,
               activation_scale_optional,
               weight_scale_optional,
               group_list_optional,
               expand_scales_optional,
               shared_expert_x_optional,
               group_ep_ptr,
               ep_world_size,
               ep_rank_id,
               moe_expert_num,
               group_tp_ptr,
               tp_world_size,
               tp_rank_id,
               expert_shard_type,
               shared_expert_num,
               shared_expert_rank_num,
               global_bs,
               kOutDtypeDefault,
               comm_quant_mode,
               kGroupListTypeDefault,
               comm_alg_ptr,
               output);

  return output;
}

}  // namespace xllm::kernel::npu
