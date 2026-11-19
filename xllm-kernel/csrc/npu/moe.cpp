// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include <glog/logging.h>
#include <torch/library.h>
#include <torch/types.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "csrc/npu/ascendc_api.h"
#include "op_plugin/utils/op_api_common_base.h"

namespace xllm_kernel {
namespace {

bool available(const AscendcApi& api) {
  return api.launch != nullptr && api.workspace != nullptr;
}

AscendcApi find_cann_api(const char* name) {
  static void* const library = dlopen("libopapi.so", RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) {
    return {};
  }
  const std::string workspace = std::string(name) + "GetWorkspaceSize";
  return {dlsym(library, name), dlsym(library, workspace.c_str())};
}

AscendcApi select_api(const char* package_name, const char* cann_name) {
  const std::string workspace = std::string(package_name) + "GetWorkspaceSize";
  const auto packaged = find_ascendc_api(package_name, workspace.c_str());
  return available(packaged) || cann_name == nullptr ? packaged
                                                     : find_cann_api(cann_name);
}

const auto& moe_apis() {
  static const std::unordered_map<std::string_view, AscendcApi> apis = {
      {"aclnnXllmKernelMoeGatingTopKHash",
       select_api("aclnnXllmKernelMoeGatingTopKHash", nullptr)},
      {"aclnnXllmKernelMoeInitRoutingV3",
       select_api("aclnnXllmKernelMoeInitRoutingV3", "aclnnMoeInitRoutingV3")},
      {"aclnnXllmKernelMoeGroupedMatmulSwigluQuant",
       select_api("aclnnXllmKernelMoeGroupedMatmulSwigluQuant", nullptr)},
      {"aclnnGroupedMatmulSwigluQuantWeightNZ",
       find_cann_api("aclnnGroupedMatmulSwigluQuantWeightNZ")},
      {"aclnnXllmKernelDequantSwigluQuantV2",
       select_api("aclnnXllmKernelDequantSwigluQuantV2",
                  "aclnnDequantSwigluQuantV2")},
      {"aclnnXllmKernelDequantSwigluQuant",
       select_api("aclnnXllmKernelDequantSwigluQuant",
                  "aclnnDequantSwigluQuant")},
  };
  return apis;
}

void GetApiFunc(const char* name,
                const char* /*workspace_name*/,
                void*& launch,
                void*& workspace) {
  const auto& api = moe_apis().at(name);
  CHECK(available(api)) << name << " is unavailable";
  launch = api.launch;
  workspace = api.workspace;
}

bool prepare_ascendc_moe() {
  const auto& apis = moe_apis();
  CHECK(available(apis.at("aclnnXllmKernelMoeGatingTopKHash")));
  CHECK(available(apis.at("aclnnXllmKernelDequantSwigluQuant")) ||
        available(apis.at("aclnnXllmKernelDequantSwigluQuantV2")));
  return available(apis.at("aclnnXllmKernelMoeInitRoutingV3")) &&
         (available(apis.at("aclnnXllmKernelMoeGroupedMatmulSwigluQuant")) ||
          available(apis.at("aclnnGroupedMatmulSwigluQuantWeightNZ")));
}

using Tensor = torch::Tensor;
using OptionalTensor = std::optional<Tensor>;

std::tuple<Tensor, Tensor, Tensor> moe_gating_top_k_hash(
    const Tensor& x,
    int64_t k,
    const OptionalTensor& bias,
    const OptionalTensor& input_ids,
    const OptionalTensor& tid2eid,
    int64_t k_group,
    int64_t group_count,
    double routed_scaling_factor,
    double eps,
    int64_t group_select_mode,
    int64_t renorm,
    int64_t norm_type,
    bool out_flag) {
  CHECK_GE(x.dim(), 1);
  CHECK_GT(k, 0);
  CHECK_LE(k, x.size(-1));
  CHECK(x.scalar_type() == torch::kFloat32 ||
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16);
  CHECK(!bias || bias->scalar_type() == x.scalar_type());
  const torch::DeviceGuard guard(x.device());
  auto y_shape = x.sizes().vec();
  y_shape.back() = k;
  auto y = torch::empty(y_shape, x.options());
  auto expert_idx = torch::empty(y_shape, x.options().dtype(torch::kInt32));
  auto out = torch::empty(x.sizes(), x.options().dtype(torch::kFloat32));

  EXEC_NPU_CMD_EXT(aclnnXllmKernelMoeGatingTopKHash,
                   x,
                   bias,
                   input_ids,
                   tid2eid,
                   k,
                   k_group,
                   group_count,
                   group_select_mode,
                   renorm,
                   norm_type,
                   out_flag,
                   routed_scaling_factor,
                   eps,
                   y,
                   expert_idx,
                   out);
  return {y, expert_idx, out};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> moe_init_routing_v3(
    const Tensor& x,
    const Tensor& expert_idx,
    int64_t active_num,
    int64_t expert_num,
    torch::IntArrayRef active_expert_range,
    int64_t quant_mode) {
  CHECK(x.dim() == 2 && expert_idx.dim() == 2 &&
        expert_idx.size(0) == x.size(0));
  CHECK_EQ(active_expert_range.size(), 2);
  CHECK_GE(active_expert_range[0], 0);
  CHECK_LE(active_expert_range[0], active_expert_range[1]);
  CHECK_LE(active_expert_range[1], expert_num);
  CHECK(quant_mode == -1 || quant_mode == 1);

  const torch::DeviceGuard guard(x.device());
  const int64_t token_count = x.size(0) * expert_idx.size(1);
  const int64_t output_tokens =
      active_num > 0 ? std::min(token_count, active_num) : token_count;
  const torch::ScalarType dtype =
      quant_mode == -1 ? x.scalar_type() : torch::kInt8;
  auto expanded_x =
      torch::empty({output_tokens, x.size(1)}, x.options().dtype(dtype));
  auto expanded_row_idx =
      torch::empty({token_count}, expert_idx.options().dtype(torch::kInt32));
  auto group_list =
      torch::empty({expert_num, 2}, x.options().dtype(torch::kInt64));
  auto expanded_scale =
      torch::empty({output_tokens}, x.options().dtype(torch::kFloat32));
  const OptionalTensor absent;
  constexpr int64_t kExpertCapacity = -1;
  constexpr int64_t kDropPadMode = 0;
  constexpr int64_t kExpertTokensNumType = 2;
  constexpr bool kExpertTokensNumFlag = true;
  constexpr int64_t kGatherRowIdxType = 0;
  EXEC_NPU_CMD_EXT(aclnnXllmKernelMoeInitRoutingV3,
                   x,
                   expert_idx,
                   /*scale=*/absent,
                   /*offset=*/absent,
                   output_tokens,
                   kExpertCapacity,
                   expert_num,
                   kDropPadMode,
                   kExpertTokensNumType,
                   kExpertTokensNumFlag,
                   quant_mode,
                   active_expert_range,
                   kGatherRowIdxType,
                   expanded_x,
                   expanded_row_idx,
                   group_list,
                   expanded_scale);
  return {expanded_x, expanded_row_idx, group_list, expanded_scale};
}

std::tuple<Tensor, Tensor> moe_grouped_matmul_swiglu_quant(
    const Tensor& x,
    const Tensor& weight,
    const Tensor& weight_scale,
    const Tensor& x_scale,
    const Tensor& group_list) {
  CHECK(x.dim() == 2 && weight.dim() == 3 && weight_scale.dim() == 2 &&
        x_scale.dim() == 1);
  CHECK(group_list.dim() == 2 && group_list.size(1) == 2);
  CHECK_EQ(weight.size(2) % 2, 0);
  const torch::DeviceGuard guard(x.device());
  const int64_t fused_width = weight.size(2);
  auto output = torch::empty({x.size(0), fused_width / 2},
                             x.options().dtype(torch::kInt8));
  auto output_scale =
      torch::empty({x.size(0)}, x.options().dtype(torch::kFloat32));

  if (available(moe_apis().at("aclnnXllmKernelMoeGroupedMatmulSwigluQuant"))) {
    const int64_t input_width = weight.size(1);
    CHECK(input_width % 16 == 0 && fused_width % 32 == 0)
        << "private NZ weights require K aligned to 16 and 2N aligned to 32";
    // The package ACLNN API takes the physical NZ dimensions.
    const std::vector<int64_t> shape = {
        weight.size(0), fused_width / 32, input_width / 16, 16, 32};
    const std::vector<int64_t> stride = {
        fused_width * input_width, input_width * 32, 16 * 32, 32, 1};
    const auto physical_weight =
        weight.as_strided(shape, stride, weight.storage_offset());
    EXEC_NPU_CMD_EXT(aclnnXllmKernelMoeGroupedMatmulSwigluQuant,
                     x,
                     physical_weight,
                     weight_scale,
                     x_scale,
                     group_list,
                     output,
                     output_scale);
  } else {
    // The CANN API consumes cumulative counts, while V3 exposes expert/count
    // pairs.
    const auto cumulative = torch::cumsum(group_list.select(1, 1), 0);
    const OptionalTensor absent;
    auto output_offset = torch::empty({}, output_scale.options());
    EXEC_NPU_CMD_EXT(aclnnGroupedMatmulSwigluQuantWeightNZ,
                     x,
                     weight,
                     /*bias=*/absent,
                     /*offset=*/absent,
                     weight_scale,
                     x_scale,
                     cumulative,
                     output,
                     output_scale,
                     output_offset);
  }
  return {output, output_scale};
}

std::tuple<Tensor, Tensor> dequant_swiglu_quant(
    const Tensor& x,
    const OptionalTensor& weight_scale,
    const OptionalTensor& activation_scale,
    const OptionalTensor& bias,
    const OptionalTensor& quant_scale,
    const OptionalTensor& quant_offset,
    const OptionalTensor& group_index,
    bool activate_left,
    int64_t quant_mode,
    int64_t swiglu_mode,
    double clamp_limit,
    double glu_alpha,
    double glu_bias) {
  CHECK(x.dim() > 1 && x.size(-1) % 2 == 0);
  CHECK(quant_mode == 0 || quant_mode == 1);
  CHECK(swiglu_mode == 0 || swiglu_mode == 1);
  CHECK(std::isfinite(clamp_limit) && clamp_limit >= 0);
  CHECK(std::isfinite(glu_alpha) && std::isfinite(glu_bias));

  const torch::DeviceGuard guard(x.device());
  auto shape = x.sizes().vec();
  shape.back() /= 2;
  auto y = torch::empty(shape, x.options().dtype(torch::kInt8));
  shape.pop_back();
  auto scale = torch::empty(shape, x.options().dtype(torch::kFloat32));
  // ACLNN's string attributes are read-only, including on the task queue.
  char* quant_mode_arg =
      const_cast<char*>(quant_mode == 1 ? "dynamic" : "static");

  if (available(moe_apis().at("aclnnXllmKernelDequantSwigluQuantV2"))) {
    char* round_mode = const_cast<char*>("rint");
    constexpr int64_t kDstTypeInt8 = 2;
    constexpr int64_t kActivateDim = -1;
    EXEC_NPU_CMD_EXT(aclnnXllmKernelDequantSwigluQuantV2,
                     x,
                     weight_scale,
                     activation_scale,
                     bias,
                     quant_scale,
                     quant_offset,
                     group_index,
                     activate_left,
                     quant_mode_arg,
                     kDstTypeInt8,
                     round_mode,
                     kActivateDim,
                     swiglu_mode,
                     clamp_limit,
                     glu_alpha,
                     glu_bias,
                     y,
                     scale);
  } else {
    CHECK_EQ(swiglu_mode, 0)
        << "SwiGLU mode 1 requires ACLNN DequantSwigluQuantV2";
    EXEC_NPU_CMD_EXT(aclnnXllmKernelDequantSwigluQuant,
                     x,
                     weight_scale,
                     activation_scale,
                     bias,
                     quant_scale,
                     quant_offset,
                     group_index,
                     activate_left,
                     quant_mode_arg,
                     y,
                     scale);
  }
  return {y, scale};
}

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("prepare_ascendc_moe() -> bool", TORCH_FN(prepare_ascendc_moe));
  m.def(
      "moe_gating_top_k_hash(Tensor x, int k, Tensor? bias, Tensor? input_ids, "
      "Tensor? tid2eid, int k_group, int group_count, float "
      "routed_scaling_factor, "
      "float eps, int group_select_mode, int renorm, int norm_type, bool "
      "out_flag) "
      "-> (Tensor, Tensor, Tensor)");
  m.def(
      "moe_init_routing_v3(Tensor x, Tensor expert_idx, int active_num, "
      "int expert_num, int[] active_expert_range, int quant_mode) "
      "-> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "moe_grouped_matmul_swiglu_quant(Tensor x, Tensor weight, "
      "Tensor weight_scale, Tensor x_scale, Tensor group_list) -> (Tensor, "
      "Tensor)");
  m.def(
      "dequant_swiglu_quant(Tensor x, Tensor? weight_scale, Tensor? "
      "activation_scale, "
      "Tensor? bias, Tensor? quant_scale, Tensor? quant_offset, Tensor? "
      "group_index, "
      "bool activate_left, int quant_mode, int swiglu_mode, float clamp_limit, "
      "float glu_alpha, float glu_bias) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("moe_gating_top_k_hash", TORCH_FN(moe_gating_top_k_hash));
  m.impl("moe_init_routing_v3", TORCH_FN(moe_init_routing_v3));
  m.impl("moe_grouped_matmul_swiglu_quant",
         TORCH_FN(moe_grouped_matmul_swiglu_quant));
  m.impl("dequant_swiglu_quant", TORCH_FN(dequant_swiglu_quant));
}
}  // namespace
}  // namespace xllm_kernel
