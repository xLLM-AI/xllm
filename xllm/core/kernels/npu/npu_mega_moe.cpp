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

#include <glog/logging.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/npu_ops_api.h"
#include "graph/types.h"

namespace xllm::kernel::npu {
namespace {

bool is_fractal_nz(const torch::Tensor& tensor) {
  return at_npu::native::custom_ops::get_npu_format(tensor) ==
         ACL_FORMAT_FRACTAL_NZ;
}

aclTensor* create_acl_tensor(const int64_t* view_dims,
                             uint64_t view_ndim,
                             aclDataType dtype,
                             const int64_t* strides,
                             aclFormat format,
                             const int64_t* storage_dims,
                             uint64_t storage_ndim,
                             void* data_ptr) {
  static const auto acl_create_tensor =
      aclnn::detail::get_op_api_func<aclnn::detail::AclCreateTensorFn>(
          "aclCreateTensor");
  CHECK(acl_create_tensor != nullptr)
      << "aclCreateTensor is not available in libopapi.";
  aclTensor* tensor = acl_create_tensor(view_dims,
                                        view_ndim,
                                        dtype,
                                        strides,
                                        /*offset=*/0,
                                        format,
                                        storage_dims,
                                        storage_ndim,
                                        data_ptr);
  CHECK(tensor != nullptr) << "aclCreateTensor returned nullptr.";
  return tensor;
}

aclTensor* create_packed_weight_view(const torch::Tensor& packed,
                                     int64_t expert_idx,
                                     int64_t local_expert_num,
                                     bool fractal_nz) {
  const int64_t view_dims[2] = {packed.size(1), packed.size(2)};
  const int64_t strides[2] = {view_dims[1], 1};
  const size_t expert_nbytes =
      packed.nbytes() / static_cast<size_t>(local_expert_num);
  auto* data = static_cast<uint8_t*>(packed.data_ptr()) +
               static_cast<size_t>(expert_idx) * expert_nbytes;
  if (fractal_nz) {
    const int64_t storage_dims[1] = {
        static_cast<int64_t>(expert_nbytes / packed.itemsize())};
    return create_acl_tensor(view_dims,
                             2,
                             ACL_INT8,
                             strides,
                             ACL_FORMAT_FRACTAL_NZ,
                             storage_dims,
                             1,
                             data);
  }
  return create_acl_tensor(
      view_dims, 2, ACL_INT8, strides, ACL_FORMAT_ND, view_dims, 2, data);
}

aclTensor* create_packed_scale_view(const torch::Tensor& packed,
                                    int64_t expert_idx,
                                    int64_t local_expert_num) {
  const int64_t expert_numel = packed.numel() / local_expert_num;
  const int64_t view_dims[1] = {expert_numel};
  const int64_t strides[1] = {1};
  const size_t expert_nbytes =
      packed.nbytes() / static_cast<size_t>(local_expert_num);
  auto* data = static_cast<uint8_t*>(packed.data_ptr()) +
               static_cast<size_t>(expert_idx) * expert_nbytes;
  // ATB reinterprets encoded int64 scale storage as uint64 for aclnnMegaMoe.
  return create_acl_tensor(
      view_dims, 1, ACL_UINT64, strides, ACL_FORMAT_ND, view_dims, 1, data);
}

}  // namespace

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
  CHECK(has_mega_moe()) << "aclnnMegaMoe is not available in libopapi.";
  CHECK(context.defined()) << "MegaMoe expects a defined context tensor.";
  CHECK(context.dim() == 1) << "MegaMoe expects 1D context.";
  CHECK(context.scalar_type() == torch::kInt)
      << "MegaMoe expects int32 context, got "
      << c10::toString(context.scalar_type());
  CHECK(x.dim() == 2) << "MegaMoe expects 2D x.";
  CHECK(x.scalar_type() == torch::kBFloat16)
      << "MegaMoe expects bf16 x, got " << c10::toString(x.scalar_type());
  CHECK(topk_ids.dim() == 2) << "MegaMoe expects 2D topk_ids.";
  CHECK(topk_ids.scalar_type() == torch::kInt)
      << "MegaMoe expects int32 topk_ids, got "
      << c10::toString(topk_ids.scalar_type());
  CHECK(topk_weights.dim() == 2) << "MegaMoe expects 2D topk_weights.";
  CHECK(topk_weights.scalar_type() == torch::kFloat)
      << "MegaMoe expects float32 topk_weights, got "
      << c10::toString(topk_weights.scalar_type());
  CHECK(topk_ids.sizes() == topk_weights.sizes())
      << "MegaMoe topk_ids/topk_weights shape mismatch: " << topk_ids.sizes()
      << " vs " << topk_weights.sizes();
  CHECK(topk_ids.size(0) == x.size(0))
      << "MegaMoe x/router token count mismatch: " << x.size(0) << " vs "
      << topk_ids.size(0);
  CHECK(!weight1.empty()) << "MegaMoe expects non-empty weight1.";
  CHECK(!weight2.empty()) << "MegaMoe expects non-empty weight2.";
  CHECK(weight1.size() == weight2.size())
      << "MegaMoe weight1/weight2 list size mismatch: " << weight1.size()
      << " vs " << weight2.size();
  CHECK(moe_expert_num > 0) << "MegaMoe requires moe_expert_num > 0.";
  CHECK(ep_world_size > 0) << "MegaMoe requires ep_world_size > 0.";
  CHECK(moe_expert_num % ep_world_size == 0)
      << "MegaMoe moe_expert_num must be divisible by ep_world_size: "
      << moe_expert_num << " vs " << ep_world_size;
  CHECK(ccl_buffer_size > 0) << "MegaMoe requires ccl_buffer_size > 0.";
  CHECK(activation == "swiglu")
      << "MegaMoe verified path requires swiglu activation, got " << activation;
  CHECK(rank_num_per_server > 0) << "MegaMoe requires rank_num_per_server > 0.";
  const int64_t local_expert_num = moe_expert_num / ep_world_size;
  const int64_t hidden_size = x.size(1);
  const bool use_w8a8 = dispatch_quant_mode == kMegaMoeDispatchQuantModeDynamic;
  if (!use_w8a8) {
    CHECK(static_cast<int64_t>(weight1.size()) == local_expert_num)
        << "MegaMoe A16W16 expects one weight pair per local expert: "
        << local_expert_num << ", got " << weight1.size();
    for (size_t expert = 0; expert < weight1.size(); ++expert) {
      const auto& w1 = weight1[expert];
      const auto& w2 = weight2[expert];
      CHECK(w1.dim() == 2 && w2.dim() == 2)
          << "MegaMoe A16W16 expert weights must be 2D at local expert "
          << expert << ".";
      CHECK(w1.scalar_type() == torch::kBFloat16 &&
            w2.scalar_type() == torch::kBFloat16)
          << "MegaMoe A16W16 expects bf16 weights at local expert " << expert
          << ".";
      CHECK(w1.size(0) == hidden_size)
          << "MegaMoe W1 hidden dimension mismatch at local expert " << expert
          << ": expected " << hidden_size << ", got " << w1.size(0);
      CHECK(w2.size(1) == hidden_size)
          << "MegaMoe W2 hidden dimension mismatch at local expert " << expert
          << ": expected " << hidden_size << ", got " << w2.size(1);
      CHECK(w1.size(1) == 2 * w2.size(0))
          << "MegaMoe W1/W2 intermediate dimension mismatch at local "
             "expert "
          << expert << ": W1 columns " << w1.size(1) << ", W2 rows "
          << w2.size(0);
    }
  } else {
    CHECK(weight1.size() == 1 && weight2.size() == 1)
        << "MegaMoe W8A8 expects one packed weight tensor for each "
           "projection.";
    const auto& w1 = weight1[0];
    const auto& w2 = weight2[0];
    CHECK(w1.dim() == 3 && w2.dim() == 3)
        << "MegaMoe W8A8 packed weights must be 3D.";
    CHECK(w1.scalar_type() == torch::kChar && w2.scalar_type() == torch::kChar)
        << "MegaMoe W8A8 expects int8 packed weights.";
    CHECK(w1.size(0) == local_expert_num && w2.size(0) == local_expert_num)
        << "MegaMoe W8A8 local expert dimension mismatch.";
    CHECK(w1.size(1) == hidden_size && w2.size(2) == hidden_size)
        << "MegaMoe W8A8 hidden dimension mismatch.";
    CHECK(w1.size(2) == 2 * w2.size(1))
        << "MegaMoe W8A8 intermediate dimension mismatch.";
    CHECK(weight_scales1.has_value() && weight_scales2.has_value())
        << "MegaMoe W8A8 requires encoded weight scales.";
    CHECK(weight_scales1->size() == 1 && weight_scales2->size() == 1)
        << "MegaMoe W8A8 expects one packed scale tensor for each "
           "projection.";
    const auto& scale1 = (*weight_scales1)[0];
    const auto& scale2 = (*weight_scales2)[0];
    CHECK(scale1.scalar_type() == torch::kLong &&
          scale2.scalar_type() == torch::kLong)
        << "MegaMoe W8A8 encoded scales must use int64 storage.";
    CHECK(scale1.numel() == local_expert_num * w1.size(2))
        << "MegaMoe W8A8 weight1 scale size mismatch.";
    CHECK(scale2.numel() == local_expert_num * hidden_size)
        << "MegaMoe W8A8 weight2 scale size mismatch.";
    CHECK(dispatch_quant_out_dtype == kMegaMoeDtypeInt8)
        << "MegaMoe W8A8 dispatch output dtype must be ACL_INT8.";
    CHECK(!bias1.has_value() && !bias2.has_value())
        << "MegaMoe W8A8 path does not accept expert bias.";
    if (x_active_mask.has_value() && x_active_mask->defined()) {
      CHECK(x_active_mask->dim() == 1)
          << "MegaMoe W8A8 expects 1D x_active_mask.";
      CHECK(x_active_mask->numel() == x.size(0))
          << "MegaMoe W8A8 x_active_mask token count mismatch: expected "
          << x.size(0) << ", got " << x_active_mask->numel();
      CHECK(x_active_mask->scalar_type() == torch::kChar)
          << "MegaMoe W8A8 expects int8 x_active_mask, got "
          << c10::toString(x_active_mask->scalar_type());
    }
  }

  auto y = torch::empty_like(x);
  auto expert_token_nums =
      torch::empty({local_expert_num}, x.options().dtype(torch::kInt));

  std::string comm_alg_copy = comm_alg;
  char* comm_alg_ptr = comm_alg_copy.data();
  std::string activation_copy = activation;
  char* activation_ptr = activation_copy.data();

  const int64_t resolved_dispatch_quant_out_dtype =
      dispatch_quant_mode == kMegaMoeDispatchQuantModeNone
          ? kMegaMoeDtypeBFloat16
          : dispatch_quant_out_dtype;

  std::array<aclTensorList*, 4> acl_lists{};
  auto& [acl_weight1, acl_weight2, acl_scales1, acl_scales2] = acl_lists;
  if (use_w8a8) {
    const auto& packed_w1 = weight1[0];
    const auto& packed_w2 = weight2[0];
    const auto& packed_s1 = (*weight_scales1)[0];
    const auto& packed_s2 = (*weight_scales2)[0];
    using TensorViews = std::vector<const aclTensor*>;
    TensorViews w1_views;
    TensorViews w2_views;
    TensorViews s1_views;
    TensorViews s2_views;
    w1_views.reserve(local_expert_num);
    w2_views.reserve(local_expert_num);
    s1_views.reserve(local_expert_num);
    s2_views.reserve(local_expert_num);
    const bool w1_fractal_nz = is_fractal_nz(packed_w1);
    const bool w2_fractal_nz = is_fractal_nz(packed_w2);
    for (int64_t expert = 0; expert < local_expert_num; ++expert) {
      w1_views.emplace_back(create_packed_weight_view(
          packed_w1, expert, local_expert_num, w1_fractal_nz));
      w2_views.emplace_back(create_packed_weight_view(
          packed_w2, expert, local_expert_num, w2_fractal_nz));
      s1_views.emplace_back(
          create_packed_scale_view(packed_s1, expert, local_expert_num));
      s2_views.emplace_back(
          create_packed_scale_view(packed_s2, expert, local_expert_num));
    }
    static const auto create_list =
        aclnn::detail::get_op_api_func<aclnn::detail::AclCreateTensorListFn>(
            "aclCreateTensorList");
    CHECK(create_list != nullptr)
        << "aclCreateTensorList is not available in libopapi.";
    const auto checked_create_list = [](const auto& tensors) {
      aclTensorList* list = create_list(tensors.data(), tensors.size());
      CHECK(list != nullptr) << "aclCreateTensorList returned nullptr.";
      return list;
    };
    acl_weight1 = checked_create_list(w1_views);
    acl_weight2 = checked_create_list(w2_views);
    acl_scales1 = checked_create_list(s1_views);
    acl_scales2 = checked_create_list(s2_views);
  } else {
    acl_weight1 = aclnn::detail::convert_type(weight1);
    acl_weight2 = aclnn::detail::convert_type(weight2);
    acl_scales1 = aclnn::detail::convert_type(weight_scales1);
    acl_scales2 = aclnn::detail::convert_type(weight_scales2);
  }

  EXEC_NPU_CMD(aclnnMegaMoe,
               context,
               x,
               topk_ids,
               topk_weights,
               acl_weight1,
               acl_weight2,
               acl_scales1,
               acl_scales2,
               bias1,
               bias2,
               x_active_mask,
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
               y,
               expert_token_nums);

  return std::make_tuple(y, expert_token_nums);
}

}  // namespace xllm::kernel::npu
