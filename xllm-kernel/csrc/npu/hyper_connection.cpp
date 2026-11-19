// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include <acl/acl.h>
#include <glog/logging.h>
#include <torch/library.h>
#include <torch/types.h>

#include <array>
#include <cmath>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>

#include "csrc/npu/ascendc_api.h"
#include "op_plugin/utils/op_api_common_base.h"

namespace xllm_kernel {
namespace {

using PreOutput = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;

const AscendcApi& hc_pre_va_api() {
  static const auto api = find_ascendc_api(
      "aclnnXllmKernelHcPreVa", "aclnnXllmKernelHcPreVaGetWorkspaceSize");
  return api;
}

const auto& hc_apis() {
  static const std::array apis = {
      std::pair{"aclnnXllmKernelHcPreInvRms",
                load_ascendc_api("aclnnXllmKernelHcPreInvRms",
                                 "aclnnXllmKernelHcPreInvRmsGetWorkspaceSize")},
      std::pair{
          "aclnnXllmKernelHcPreSinkhorn",
          load_ascendc_api("aclnnXllmKernelHcPreSinkhorn",
                           "aclnnXllmKernelHcPreSinkhornGetWorkspaceSize")},
      std::pair{"aclnnXllmKernelHcPost",
                load_ascendc_api("aclnnXllmKernelHcPost",
                                 "aclnnXllmKernelHcPostGetWorkspaceSize")},
      std::pair{"aclnnXllmKernelHcPreVa", hc_pre_va_api()},
  };
  return apis;
}

void GetApiFunc(const char* launch_name,
                const char* /*workspace_name*/,
                void*& launch,
                void*& workspace) {
  for (const auto& [name, api] : hc_apis()) {
    if (std::string_view(launch_name) == name) {
      launch = api.launch;
      workspace = api.workspace;
      return;
    }
  }
  LOG(FATAL) << "Unknown HyperConnection ACLNN entry point: " << launch_name;
}

bool has_hc_pre_va() {
  const auto& api = hc_pre_va_api();
  if (!api.launch || !api.workspace) {
    return false;
  }
  static const bool is_a3 = [] {
    const char* soc = aclrtGetSocName();
    return soc != nullptr && std::string_view(soc).find("Ascend910_93") == 0;
  }();
  return is_a3;
}

void prepare_ascendc_hyper_connection() { hc_apis(); }

PreOutput hc_pre(const torch::Tensor& x,
                 const torch::Tensor& hc_fn,
                 const torch::Tensor& hc_scale,
                 const torch::Tensor& hc_base,
                 int64_t hc_mult,
                 int64_t hc_sinkhorn_iters,
                 double norm_eps,
                 double hc_eps) {
  CHECK((x.dim() == 3 || x.dim() == 4) && x.numel() > 0)
      << "hc_pre requires nonempty [T, HC, D] or [B, S, HC, D] input";
  CHECK_GT(hc_mult, 0);
  CHECK_EQ(x.size(-2), hc_mult);
  CHECK_GE(hc_sinkhorn_iters, 0);
  CHECK(std::isfinite(norm_eps) && norm_eps > 0 && std::isfinite(hc_eps) &&
        hc_eps > 0);
  const int64_t mix_count = (2 + hc_mult) * hc_mult;
  CHECK(hc_fn.dim() == 2 && hc_fn.size(0) == mix_count &&
        hc_fn.size(1) == hc_mult * x.size(-1));
  CHECK(hc_scale.dim() == 1 && hc_scale.numel() == 3 && hc_base.dim() == 1 &&
        hc_base.numel() == mix_count);
  CHECK(hc_scale.scalar_type() == torch::kFloat32 &&
        hc_base.scalar_type() == torch::kFloat32);
  CHECK(hc_fn.device() == x.device() && hc_scale.device() == x.device() &&
        hc_base.device() == x.device());

  const torch::DeviceGuard guard(x.device());
  const auto x_bf16 = x.to(torch::kBFloat16);
  auto output_shape = x.sizes().vec();
  output_shape.erase(output_shape.end() - 2);
  auto post_shape = x.sizes().vec();
  post_shape.pop_back();
  auto comb_shape = post_shape;
  comb_shape.push_back(hc_mult);
  auto output = torch::empty(output_shape, x_bf16.options());
  const auto fp32_options = x.options().dtype(torch::kFloat32);
  auto post = torch::empty(post_shape, fp32_options);
  auto comb = torch::empty(comb_shape, fp32_options);
  if (has_hc_pre_va() && hc_mult == 4 && hc_sinkhorn_iters > 0 &&
      hc_fn.scalar_type() == torch::kFloat32 && x_bf16.is_contiguous() &&
      hc_fn.is_contiguous() && hc_scale.is_contiguous() &&
      hc_base.is_contiguous()) {
    const std::optional<torch::Tensor> absent;
    EXEC_NPU_CMD_EXT(aclnnXllmKernelHcPreVa,
                     x_bf16,
                     hc_fn,
                     hc_scale,
                     hc_base,
                     /*pre_mix=*/absent,
                     hc_mult,
                     hc_sinkhorn_iters,
                     hc_eps,
                     norm_eps,
                     output,
                     post,
                     comb,
                     /*pre=*/absent);
  } else {
    auto rms_shape = post_shape;
    rms_shape.back() = 1;
    auto inv_rms = torch::empty(rms_shape, fp32_options);
    const auto input = x_bf16.contiguous();
    EXEC_NPU_CMD_EXT(aclnnXllmKernelHcPreInvRms, input, norm_eps, inv_rms);

    // Match the staged projection's BF16 inputs and FP32 Sinkhorn logits.
    const auto mixes =
        torch::linear(input.flatten(-2), hc_fn.to(torch::kBFloat16))
            .to(torch::kFloat32);
    const auto scale = hc_scale.contiguous();
    const auto base = hc_base.contiguous();
    EXEC_NPU_CMD_EXT(aclnnXllmKernelHcPreSinkhorn,
                     mixes,
                     inv_rms,
                     scale,
                     base,
                     input,
                     hc_mult,
                     hc_sinkhorn_iters,
                     hc_eps,
                     output,
                     post,
                     comb);
  }
  return {output.to(x.scalar_type()), post, comb};
}

torch::Tensor hc_post(const torch::Tensor& x,
                      const torch::Tensor& residual,
                      const torch::Tensor& post,
                      const torch::Tensor& comb) {
  CHECK(x.dim() == 3 && x.numel() > 0)
      << "hc_post requires nonempty [B, S, D] input";
  CHECK_EQ(residual.dim(), 4);
  const int64_t hc_mult = residual.size(-2);
  CHECK_GT(hc_mult, 0);
  const int64_t batch = x.size(0);
  const int64_t sequence = x.size(1);
  CHECK(residual.sizes() ==
        torch::IntArrayRef({batch, sequence, hc_mult, x.size(-1)}));
  CHECK(post.sizes() == torch::IntArrayRef({batch, sequence, hc_mult}));
  CHECK(comb.sizes() ==
        torch::IntArrayRef({batch, sequence, hc_mult, hc_mult}));
  CHECK(x.scalar_type() == residual.scalar_type() &&
        post.scalar_type() == comb.scalar_type());
  CHECK(residual.device() == x.device() && post.device() == x.device() &&
        comb.device() == x.device());

  const torch::DeviceGuard guard(x.device());
  auto output = torch::empty(residual.sizes(), residual.options());
  const auto input = x.contiguous();
  const auto residual_nd = residual.contiguous();
  const auto post_nd = post.contiguous();
  const auto comb_nd = comb.contiguous();
  EXEC_NPU_CMD_EXT(
      aclnnXllmKernelHcPost, input, residual_nd, post_nd, comb_nd, output);
  return output;
}

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("prepare_ascendc_hyper_connection() -> ()",
        TORCH_FN(prepare_ascendc_hyper_connection));
  m.def(
      "hc_pre(Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, "
      "int hc_mult, int hc_sinkhorn_iters, float norm_eps, float hc_eps) -> "
      "(Tensor, Tensor, Tensor)");
  m.def(
      "hc_post(Tensor x, Tensor residual, Tensor post, Tensor comb) -> Tensor");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("hc_pre", TORCH_FN(hc_pre));
  m.impl("hc_post", TORCH_FN(hc_post));
}

}  // namespace
}  // namespace xllm_kernel
