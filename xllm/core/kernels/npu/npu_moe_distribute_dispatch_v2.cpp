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

#include <algorithm>
#include <optional>
#include <string>
#include <tuple>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"

namespace xllm::kernel::npu {
namespace {

constexpr int64_t kNoQuantMode = 0;

int64_t resolve_local_expert_num(int64_t ep_world_size,
                                 int64_t moe_expert_num,
                                 int64_t shared_expert_rank_num) {
  const int64_t routed_ep_size = ep_world_size - shared_expert_rank_num;
  CHECK_GT(routed_ep_size, 0)
      << "invalid shared_expert_rank_num=" << shared_expert_rank_num
      << " for ep_world_size=" << ep_world_size;
  CHECK_EQ(moe_expert_num % routed_ep_size, 0)
      << "moe_expert_num must be divisible by routed EP size, got "
      << moe_expert_num << " / " << routed_ep_size;
  return moe_expert_num / routed_ep_size;
}

int64_t resolve_dispatch_capacity(int64_t local_bs,
                                  int64_t topk,
                                  int64_t local_expert_num,
                                  int64_t ep_world_size,
                                  int64_t global_bs) {
  const int64_t global_bs_real =
      global_bs == 0 ? local_bs * ep_world_size : global_bs;
  return global_bs_real * std::min(local_expert_num, topk);
}

bool has_dispatch_v2() {
  static const bool is_available =
      aclnn::detail::get_op_api_func_addr(
          "aclnnMoeDistributeDispatchV2GetWorkspaceSize") != nullptr &&
      aclnn::detail::get_op_api_func_addr("aclnnMoeDistributeDispatchV2") !=
          nullptr;
  return is_available;
}

}  // namespace

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
apply_npu_moe_distribute_dispatch_v2(
    const torch::Tensor& x,
    const torch::Tensor& expert_ids,
    const std::optional<torch::Tensor>& expert_scales,
    const std::optional<torch::Tensor>& x_active_mask,
    const std::optional<torch::Tensor>& scales,
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
    int64_t quant_mode,
    int64_t global_bs,
    int64_t expert_token_nums_type,
    const std::string& comm_alg) {
  CHECK(has_dispatch_v2())
      << "aclnnMoeDistributeDispatchV2 is not available in libopapi.";
  CHECK_EQ(x.dim(), 2) << "MoeDistributeDispatchV2 expects 2D x.";
  CHECK(x.scalar_type() == torch::kHalf || x.scalar_type() == torch::kBFloat16)
      << "MoeDistributeDispatchV2 only supports FP16/BF16 x, got "
      << c10::toString(x.scalar_type());
  CHECK_EQ(expert_ids.dim(), 2)
      << "MoeDistributeDispatchV2 expects 2D expert_ids.";
  CHECK_EQ(expert_ids.scalar_type(), torch::kInt32)
      << "MoeDistributeDispatchV2 expects int32 expert_ids, got "
      << c10::toString(expert_ids.scalar_type());
  CHECK(!group_ep.empty())
      << "MoeDistributeDispatchV2 requires non-empty EP group name.";
  CHECK_GT(ep_world_size, 1)
      << "MoeDistributeDispatchV2 requires ep_world_size > 1.";
  CHECK(ep_rank_id >= 0 && ep_rank_id < ep_world_size)
      << "invalid EP rank " << ep_rank_id << " for ep_world_size "
      << ep_world_size;
  CHECK(tp_world_size == 0 || tp_world_size == 1 || tp_world_size == 2)
      << "tp_world_size must be 0, 1, or 2, got " << tp_world_size;
  CHECK_EQ(quant_mode, kNoQuantMode)
      << "xLLM currently uses MoeDistributeDispatchV2 only in "
         "non-quantized communication mode.";

  const int64_t local_bs = x.size(0);
  const int64_t hidden = x.size(1);
  const int64_t topk = expert_ids.size(1);
  CHECK_GT(topk, 0) << "MoeDistributeDispatchV2 requires topk > 0.";
  const int64_t local_expert_num = resolve_local_expert_num(
      ep_world_size, moe_expert_num, shared_expert_rank_num);
  const int64_t capacity = resolve_dispatch_capacity(
      local_bs, topk, local_expert_num, ep_world_size, global_bs);
  const int64_t tp_factor = std::max<int64_t>(tp_world_size, 1);
  int64_t ep_recv_count_num =
      tp_world_size == 2 ? ep_world_size * local_expert_num * tp_world_size
                         : ep_world_size * local_expert_num;
  int64_t assist_info_num = std::max(local_bs * topk, capacity * 128);
  const bool has_expert_scales =
      expert_scales.has_value() && expert_scales->defined();
  if (has_expert_scales) {
    CHECK_EQ(expert_scales->dim(), 2)
        << "MoeDistributeDispatchV2 expects 2D expert_scales.";
    CHECK_EQ(expert_scales->scalar_type(), torch::kFloat32)
        << "MoeDistributeDispatchV2 expects float32 expert_scales, got "
        << c10::toString(expert_scales->scalar_type());
    const int64_t global_bs_real =
        global_bs == 0 ? local_bs * ep_world_size : global_bs;
    const int64_t scale_info_num =
        global_bs_real * 2 * topk * (ep_world_size / 8);
    ep_recv_count_num += scale_info_num;
    assist_info_num = std::max(assist_info_num, scale_info_num);
  }

  torch::Tensor expand_x =
      torch::empty({capacity * tp_factor, hidden}, x.options());
  torch::Tensor dynamic_scales =
      torch::empty({0}, x.options().dtype(torch::kFloat32));
  auto assist_info_for_combine =
      torch::empty({assist_info_num}, x.options().dtype(torch::kInt32));
  auto expert_token_nums =
      torch::empty({local_expert_num}, x.options().dtype(torch::kInt64));
  auto ep_recv_counts =
      torch::empty({ep_recv_count_num}, x.options().dtype(torch::kInt32));
  auto tp_recv_counts =
      torch::empty({tp_world_size}, x.options().dtype(torch::kInt32));
  auto expand_scales =
      has_expert_scales
          ? torch::empty({capacity}, x.options().dtype(torch::kFloat32))
          : torch::empty({0}, x.options().dtype(torch::kFloat32));

  auto scales_optional = scales.has_value() && scales->defined()
                             ? c10::optional<at::Tensor>(scales.value())
                             : c10::nullopt;
  auto x_active_mask_optional =
      x_active_mask.has_value() && x_active_mask->defined()
          ? c10::optional<at::Tensor>(x_active_mask.value())
          : c10::nullopt;
  auto expert_scales_optional =
      expert_scales.has_value() && expert_scales->defined()
          ? c10::optional<at::Tensor>(expert_scales.value())
          : c10::nullopt;

  std::string group_ep_copy = group_ep;
  char* group_ep_ptr = group_ep_copy.data();
  std::string group_tp_copy = group_tp;
  char* group_tp_ptr = group_tp_copy.data();
  std::string comm_alg_copy = comm_alg;
  char* comm_alg_ptr = comm_alg_copy.data();

  EXEC_NPU_CMD(aclnnMoeDistributeDispatchV2,
               x,
               expert_ids,
               scales_optional,
               x_active_mask_optional,
               expert_scales_optional,
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
               quant_mode,
               global_bs,
               expert_token_nums_type,
               comm_alg_ptr,
               expand_x,
               dynamic_scales,
               assist_info_for_combine,
               expert_token_nums,
               ep_recv_counts,
               tp_recv_counts,
               expand_scales);

  return std::make_tuple(expand_x,
                         dynamic_scales,
                         assist_info_for_combine,
                         expert_token_nums,
                         ep_recv_counts,
                         tp_recv_counts,
                         expand_scales);
}

bool has_moe_distribute_dispatch_combine_v2() {
  return has_dispatch_v2() &&
         aclnn::detail::get_op_api_func_addr(
             "aclnnMoeDistributeCombineV2GetWorkspaceSize") != nullptr &&
         aclnn::detail::get_op_api_func_addr("aclnnMoeDistributeCombineV2") !=
             nullptr;
}

}  // namespace xllm::kernel::npu
