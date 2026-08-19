/* Copyright 2025-2026 The xLLM Authors.

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

#include <limits>
#include <optional>
#include <string>
#include <tuple>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "graph/types.h"

namespace xllm::kernel::npu {

bool has_mega_moe() {
  static const bool is_available =
      aclnn::detail::get_op_api_func_addr("aclnnMegaMoeGetWorkspaceSize") !=
          nullptr &&
      aclnn::detail::get_op_api_func_addr("aclnnMegaMoe") != nullptr;
  return is_available;
}

// To use the MegaMoe operator, you need to download CANN version 9.1.0 and the
// corresponding ops package. Download path:
// https://www.hiascend.com/cann/download?versionId=767&ids=d802%2Ch0501%2Ch0601%2Ch0703
// If you are using CANN 9.0.0 and an A3 environment, you can upgrade only the
// ops package to CANN 9.1.0.
std::tuple<torch::Tensor, torch::Tensor> apply_npu_mega_moe(
    const torch::Tensor& context,
    const torch::Tensor& x,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    const torch::TensorList weight1,
    const torch::TensorList weight2,
    int64_t moe_expert_num,
    int64_t ep_world_size,
    int64_t ccl_buffer_size,
    const std::optional<torch::TensorList>& weight_scales1,
    const std::optional<torch::TensorList>& weight_scales2,
    const std::optional<torch::TensorList>& bias1,
    const std::optional<torch::TensorList>& bias2,
    const std::optional<torch::Tensor>& x_active_mask,
    int64_t max_recv_token_num,
    int64_t dispatch_quant_mode,
    int64_t combine_quant_mode,
    const std::string& comm_alg,
    int64_t num_max_tokens_per_rank,
    const std::string& activation,
    float activation_clamp,
    int64_t dispatch_quant_out_dtype,
    int64_t topo_type,
    int64_t rank_num_per_server) {
  TORCH_CHECK(has_mega_moe(), "aclnnMegaMoe is not available in libopapi.");
  TORCH_CHECK(context.defined(), "MegaMoe expects a defined context tensor.");
  TORCH_CHECK(context.dim() == 1, "MegaMoe expects 1D context.");
  TORCH_CHECK(context.scalar_type() == at::kInt,
              "MegaMoe expects int32 context, got ",
              c10::toString(context.scalar_type()));
  TORCH_CHECK(x.dim() == 2, "MegaMoe expects 2D x.");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16,
              "MegaMoe A16W16 path expects bf16 x, got ",
              c10::toString(x.scalar_type()));
  TORCH_CHECK(topk_ids.dim() == 2, "MegaMoe expects 2D topk_ids.");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt,
              "MegaMoe expects int32 topk_ids, got ",
              c10::toString(topk_ids.scalar_type()));
  TORCH_CHECK(topk_weights.dim() == 2, "MegaMoe expects 2D topk_weights.");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat,
              "MegaMoe expects float32 topk_weights, got ",
              c10::toString(topk_weights.scalar_type()));
  TORCH_CHECK(topk_ids.sizes() == topk_weights.sizes(),
              "MegaMoe topk_ids/topk_weights shape mismatch: ",
              topk_ids.sizes(),
              " vs ",
              topk_weights.sizes());
  TORCH_CHECK(topk_ids.size(0) == x.size(0),
              "MegaMoe x/router token count mismatch: ",
              x.size(0),
              " vs ",
              topk_ids.size(0));
  TORCH_CHECK(!weight1.empty(), "MegaMoe expects non-empty weight1.");
  TORCH_CHECK(!weight2.empty(), "MegaMoe expects non-empty weight2.");
  TORCH_CHECK(weight1.size() == weight2.size(),
              "MegaMoe weight1/weight2 list size mismatch: ",
              weight1.size(),
              " vs ",
              weight2.size());
  TORCH_CHECK(moe_expert_num > 0, "MegaMoe requires moe_expert_num > 0.");
  TORCH_CHECK(ep_world_size > 0, "MegaMoe requires ep_world_size > 0.");
  TORCH_CHECK(moe_expert_num % ep_world_size == 0,
              "MegaMoe moe_expert_num must be divisible by ep_world_size: ",
              moe_expert_num,
              " vs ",
              ep_world_size);
  TORCH_CHECK(ccl_buffer_size > 0, "MegaMoe requires ccl_buffer_size > 0.");
  TORCH_CHECK(activation == "swiglu",
              "MegaMoe verified path requires swiglu activation, got ",
              activation);
  TORCH_CHECK(rank_num_per_server > 0,
              "MegaMoe requires rank_num_per_server > 0.");

  const int64_t local_expert_num = moe_expert_num / ep_world_size;
  TORCH_CHECK(static_cast<int64_t>(weight1.size()) == local_expert_num,
              "MegaMoe expects one weight pair per local expert: ",
              local_expert_num,
              ", got ",
              weight1.size());

  const int64_t hidden_size = x.size(1);
  for (size_t expert = 0; expert < weight1.size(); ++expert) {
    const auto& w1 = weight1[expert];
    const auto& w2 = weight2[expert];
    TORCH_CHECK(w1.dim() == 2 && w2.dim() == 2,
                "MegaMoe expert weights must be 2D at local expert ",
                expert,
                ".");
    TORCH_CHECK(
        w1.scalar_type() == at::kBFloat16 && w2.scalar_type() == at::kBFloat16,
        "MegaMoe A16W16 expects bf16 weights at local expert ",
        expert,
        ".");
    TORCH_CHECK(w1.size(0) == hidden_size,
                "MegaMoe W1 hidden dimension mismatch at local expert ",
                expert,
                ": expected ",
                hidden_size,
                ", got ",
                w1.size(0));
    TORCH_CHECK(w2.size(1) == hidden_size,
                "MegaMoe W2 hidden dimension mismatch at local expert ",
                expert,
                ": expected ",
                hidden_size,
                ", got ",
                w2.size(1));
    TORCH_CHECK(w1.size(1) == 2 * w2.size(0),
                "MegaMoe W1/W2 intermediate dimension mismatch at local "
                "expert ",
                expert,
                ": W1 columns ",
                w1.size(1),
                ", W2 rows ",
                w2.size(0));
  }

  auto y = at::empty_like(x);
  auto expert_token_nums =
      at::empty({local_expert_num}, x.options().dtype(at::kInt));

  std::string comm_alg_copy = comm_alg;
  char* comm_alg_ptr = comm_alg_copy.data();
  std::string activation_copy = activation;
  char* activation_ptr = activation_copy.data();

  const int64_t resolved_dispatch_quant_out_dtype =
      dispatch_quant_mode == 0  // DISPATCH_QUANT_MODE_NO_QUANT (A16W16)
          ? 28
          : dispatch_quant_out_dtype;

  const std::optional<at::TensorList> no_tensor_list;
  const int64_t activation_out_dtype_val = 28;
  const bool transpose_weight1 = false;
  const bool transpose_weight2 = false;
  const int64_t weight1_interleave = 0;
  const int64_t topkWeightsType = 0;

  EXEC_NPU_CMD(aclnnMegaMoe,
               context,
               x,
               topk_ids,
               topk_weights,
               weight1,
               weight2,
               weight_scales1,  // weight_scales1 (nullptr on A16W16)
               weight_scales2,  // weight_scales2 (nullptr on A16W16)
               bias1,           // bias1 (nullptr on A16W16)
               bias2,           // bias2 (nullptr on A16W16)
               x_active_mask,
               //    no_tensor_list,    // shared_weight1 (no shared expert)
               //    no_tensor_list,    // shared_weight2
               //    no_tensor_list,    // shared_weight_scales1
               //    no_tensor_list,    // shared_weight_scales2
               //    no_tensor_list,    // shared_bias1
               //    no_tensor_list,    // shared_bias2
               moe_expert_num,
               ep_world_size,
               ccl_buffer_size,
               max_recv_token_num,
               dispatch_quant_mode,
               resolved_dispatch_quant_out_dtype,
               combine_quant_mode,
               comm_alg_ptr,
               num_max_tokens_per_rank,
               activation_ptr,
               activation_clamp,
               //    topo_type,
               //    rank_num_per_server,
               //    topkWeightsType,
               y,
               expert_token_nums);

  return std::make_tuple(y, expert_token_nums);
}

}  // namespace xllm::kernel::npu
