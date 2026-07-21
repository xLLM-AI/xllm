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

#include <cstdlib>
#include <optional>
#include <string>
#include <tuple>

#include "core/kernels/npu/mega_moe_acl_contract.h"
#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/npu_ops_api.h"

namespace xllm::kernel::npu {
namespace {

std::string symbol_library_path(void* symbol) {
  if (symbol == nullptr) {
    return {};
  }
  Dl_info info{};
  if (dladdr(symbol, &info) == 0 || info.dli_fname == nullptr) {
    return {};
  }
  return aclnn::detail::real_path(info.dli_fname);
}

std::string expected_custom_mega_moe_library() {
  const char* custom_opp_path = std::getenv("ASCEND_CUSTOM_OPP_PATH");
  if (custom_opp_path == nullptr) {
    return {};
  }

  std::string root(custom_opp_path);
  // MegaMoe's ABI differs from the stock CANN 9.1 op. The validated MegaMoe
  // vendor must be first, while standard companion vendors may follow for
  // other Qwen3.5 ops. The resolved MegaMoe symbols are still required to
  // come from this first vendor's single libcust_opapi.so.
  const size_t separator = root.find(':');
  if (separator != std::string::npos) {
    root.resize(separator);
  }
  if (root.empty()) {
    return {};
  }
  return aclnn::detail::real_path(
      root + "/op_api/lib/" + aclnn::detail::get_cust_op_api_lib_name());
}

void validate_weight_lists(const torch::Tensor& x,
                           const torch::TensorList weight1,
                           const torch::TensorList weight2,
                           int64_t local_expert_num) {
  TORCH_CHECK(!weight1.empty(), "MegaMoe expects non-empty weight1.");
  TORCH_CHECK(!weight2.empty(), "MegaMoe expects non-empty weight2.");
  TORCH_CHECK(weight1.size() == weight2.size(),
              "MegaMoe weight1/weight2 list size mismatch: ",
              weight1.size(),
              " vs ",
              weight2.size());
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
    TORCH_CHECK(w1.scalar_type() == at::kBFloat16 &&
                    w2.scalar_type() == at::kBFloat16,
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
}

}  // namespace

MegaMoeOpApiProvenance validate_mega_moe_op_api_paths(
    const std::string& expected_library,
    const std::string& workspace_library,
    const std::string& execute_library) {
  MegaMoeOpApiProvenance provenance;
  provenance.expected_library = expected_library;
  provenance.workspace_library = workspace_library;
  provenance.execute_library = execute_library;
  provenance.same_library = !workspace_library.empty() &&
                            workspace_library == execute_library;
  provenance.compatible = !expected_library.empty() &&
                          provenance.same_library &&
                          workspace_library == expected_library;
  return provenance;
}

MegaMoeOpApiProvenance inspect_mega_moe_op_api_provenance() {
  void* workspace =
      aclnn::detail::get_op_api_func_addr("aclnnMegaMoeGetWorkspaceSize");
  void* execute = aclnn::detail::get_op_api_func_addr("aclnnMegaMoe");
  return validate_mega_moe_op_api_paths(expected_custom_mega_moe_library(),
                                        symbol_library_path(workspace),
                                        symbol_library_path(execute));
}

bool has_mega_moe() {
  static const MegaMoeOpApiProvenance provenance =
      inspect_mega_moe_op_api_provenance();
  return provenance.compatible;
}

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
  const auto provenance = inspect_mega_moe_op_api_provenance();
  TORCH_CHECK(
      provenance.compatible,
      "aclnnMegaMoe custom ABI provenance check failed. Expected both "
      "symbols in '",
      provenance.expected_library,
      "', but GetWorkspaceSize resolved to '",
      provenance.workspace_library,
      "' and execute resolved to '",
      provenance.execute_library,
      "'. Source CANN 9.1 and put the validated MegaMoe vendor first in "
      "ASCEND_CUSTOM_OPP_PATH; standard companion vendors may follow, but "
      "both MegaMoe symbols must resolve to the first vendor.");
  TORCH_CHECK(context.defined(), "MegaMoe expects a defined context tensor.");
  TORCH_CHECK(context.dim() == 1, "MegaMoe expects 1D context.");
  TORCH_CHECK(context.scalar_type() == at::kInt,
              "MegaMoe expects int32 context, got ",
              c10::toString(context.scalar_type()));
  TORCH_CHECK(x.dim() == 2, "MegaMoe expects 2D x.");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16,
              "MegaMoe verified A16W16 path expects bf16 x, got ",
              c10::toString(x.scalar_type()));
  TORCH_CHECK(topk_ids.dim() == 2, "MegaMoe expects 2D topk_ids.");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt,
              "MegaMoe expects int32 topk_ids, got ",
              c10::toString(topk_ids.scalar_type()));
  TORCH_CHECK(topk_weights.dim() == 2,
              "MegaMoe expects 2D topk_weights.");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat,
              "MegaMoe verified A3 path requires fp32 topk_weights, got ",
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
  TORCH_CHECK(moe_expert_num > 0,
              "MegaMoe requires moe_expert_num > 0.");
  TORCH_CHECK(ep_world_size > 0, "MegaMoe requires ep_world_size > 0.");
  TORCH_CHECK(moe_expert_num % ep_world_size == 0,
              "MegaMoe moe_expert_num must be divisible by ep_world_size: ",
              moe_expert_num,
              " vs ",
              ep_world_size);
  TORCH_CHECK(ccl_buffer_size > 0,
              "MegaMoe requires ccl_buffer_size > 0.");

  const int64_t local_expert_num = moe_expert_num / ep_world_size;
  validate_weight_lists(x, weight1, weight2, local_expert_num);

  TORCH_CHECK(!weight_scales1.has_value() && !weight_scales2.has_value() &&
                  !bias1.has_value() && !bias2.has_value() &&
                  !x_active_mask.has_value(),
              "MegaMoe verified A16W16 path does not accept scales, biases, "
              "or x_active_mask.");
  TORCH_CHECK(dispatch_quant_mode == 0 && combine_quant_mode == 0,
              "MegaMoe verified A16W16 path requires quant modes 0.");
  TORCH_CHECK(activation == "swiglu",
              "MegaMoe verified path requires swiglu activation, got ",
              activation);
  TORCH_CHECK(max_recv_token_num >= 0,
              "MegaMoe requires max_recv_token_num >= 0.");
  TORCH_CHECK(num_max_tokens_per_rank >= 0,
              "MegaMoe requires num_max_tokens_per_rank >= 0.");
  TORCH_CHECK(rank_num_per_server > 0,
              "MegaMoe requires rank_num_per_server > 0.");

  auto output = at::empty_like(x);
  auto expert_token_nums =
      at::empty({local_expert_num}, x.options().dtype(at::kInt));

  std::string comm_alg_copy = comm_alg;
  char* comm_alg_ptr = comm_alg_copy.data();
  std::string activation_copy = activation;
  char* activation_ptr = activation_copy.data();

  // The generic bridge provides a testable record of the deployed custom ABI
  // ordering while retaining EXEC_NPU_CMD's lvalue conversion contract.
  auto context_arg = context;
  auto x_arg = x;
  auto topk_ids_arg = topk_ids;
  auto topk_weights_arg = topk_weights;
  auto weight1_arg = weight1;
  auto weight2_arg = weight2;
  auto weight_scales1_arg = weight_scales1;
  auto weight_scales2_arg = weight_scales2;
  auto bias1_arg = bias1;
  auto bias2_arg = bias2;
  auto x_active_mask_arg = x_active_mask;
  auto execute = [&](auto&... arguments) {
    EXEC_NPU_CMD(aclnnMegaMoe, arguments...);
  };
  execute_mega_moe_acl_contract(execute,
                                context_arg,
                                x_arg,
                                topk_ids_arg,
                                topk_weights_arg,
                                weight1_arg,
                                weight2_arg,
                                weight_scales1_arg,
                                weight_scales2_arg,
                                bias1_arg,
                                bias2_arg,
                                x_active_mask_arg,
                                moe_expert_num,
                                ep_world_size,
                                ccl_buffer_size,
                                max_recv_token_num,
                                dispatch_quant_mode,
                                dispatch_quant_out_dtype,
                                combine_quant_mode,
                                comm_alg_ptr,
                                num_max_tokens_per_rank,
                                activation_ptr,
                                activation_clamp,
                                topo_type,
                                rank_num_per_server,
                                output,
                                expert_token_nums);

  return std::make_tuple(output, expert_token_nums);
}

}  // namespace xllm::kernel::npu
