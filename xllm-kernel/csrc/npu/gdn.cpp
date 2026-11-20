// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include <glog/logging.h>
#include <torch/library.h>
#include <torch/types.h>

#include <array>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "csrc/npu/ascendc_api.h"
#include "op_plugin/utils/op_api_common_base.h"

namespace xllm_kernel {
namespace {

using Tensor = torch::Tensor;
using IntArray = torch::IntArrayRef;

const auto& gdn_apis() {
  static const std::array apis = {
      std::pair{
          "aclnnXllmKernelCausalConv1d",
          load_ascendc_api("aclnnXllmKernelCausalConv1d",
                           "aclnnXllmKernelCausalConv1dGetWorkspaceSize")},
      std::pair{
          "aclnnXllmKernelCausalConv1dQkv",
          load_ascendc_api("aclnnXllmKernelCausalConv1dQkv",
                           "aclnnXllmKernelCausalConv1dQkvGetWorkspaceSize")},
      std::pair{
          "aclnnXllmKernelMegaChunkGdn",
          load_ascendc_api("aclnnXllmKernelMegaChunkGdn",
                           "aclnnXllmKernelMegaChunkGdnGetWorkspaceSize")},
  };
  return apis;
}

void GetApiFunc(const char* name,
                const char* /*workspace_name*/,
                void*& launch,
                void*& workspace) {
  for (const auto& [api_name, api] : gdn_apis()) {
    if (std::string_view(name) == api_name) {
      launch = api.launch;
      workspace = api.workspace;
      return;
    }
  }
  LOG(FATAL) << "Unknown GDN or causal convolution ACLNN entry point: " << name;
}

void prepare_ascendc_gdn() { gdn_apis(); }

Tensor causal_conv1d(const Tensor& input,
                     const Tensor& weight,
                     const Tensor& conv_state,
                     IntArray query_start_loc,
                     int64_t activation_mode,
                     int64_t run_mode) {
  CHECK(input.dim() == 2 || input.dim() == 3);
  CHECK_EQ(weight.dim(), 2);
  CHECK_EQ(conv_state.dim(), 3);
  CHECK(input.numel() > 0);
  if (input.dim() == 3) {
    CHECK(query_start_loc.empty());
    CHECK_EQ(input.size(0), conv_state.size(0));
  } else {
    CHECK_EQ(query_start_loc.size(), conv_state.size(0) + 1);
    CHECK_EQ(query_start_loc.front(), 0);
    CHECK_EQ(query_start_loc.back(), input.size(0));
  }
  CHECK_EQ(input.size(-1), weight.size(1));
  CHECK_EQ(input.size(-1), conv_state.size(2));
  CHECK_EQ(conv_state.size(1), weight.size(0) - 1);
  CHECK(activation_mode == 0 || activation_mode == 1);
  CHECK(run_mode == 0 || run_mode == 1);
  CHECK(input.scalar_type() == weight.scalar_type() &&
        input.scalar_type() == conv_state.scalar_type());
  CHECK(input.is_contiguous() && weight.is_contiguous() &&
        conv_state.is_contiguous());

  const torch::DeviceGuard guard(input.device());
  constexpr int64_t kPadSlotId = -1;
  const std::optional<Tensor> bias;
  const IntArray no_indices;
  const IntArray no_accepted_tokens;
  std::vector<int64_t> initial_state_modes;
  if (run_mode == 0) {
    initial_state_modes.assign(conv_state.size(0), 1);
  }
  const IntArray initial_state_mode(initial_state_modes);
  auto output = torch::empty(input.sizes(), input.options());
  EXEC_NPU_CMD_EXT(aclnnXllmKernelCausalConv1d,
                   input,
                   weight,
                   bias,
                   conv_state,
                   query_start_loc,
                   no_indices,
                   initial_state_mode,
                   no_accepted_tokens,
                   activation_mode,
                   kPadSlotId,
                   run_mode,
                   output);
  return output;
}

std::tuple<Tensor, Tensor, Tensor> causal_conv1d_qkv_prefill(
    const Tensor& x,
    const Tensor& weight,
    const Tensor& conv_state,
    IntArray query_start_loc,
    IntArray state_indices,
    IntArray has_initial_state,
    int64_t num_qk_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  CHECK(x.dim() == 2 && weight.dim() == 2 && conv_state.dim() == 3);
  CHECK(num_qk_heads > 0 && num_v_heads > 0);
  CHECK(head_k_dim == 128 && head_v_dim == head_k_dim);
  const int64_t qk_dim = num_qk_heads * head_k_dim;
  const int64_t v_dim = num_v_heads * head_v_dim;
  CHECK_EQ(x.size(1), 2 * qk_dim + v_dim);

  const torch::DeviceGuard guard(x.device());
  constexpr int64_t kPackedQkvActivationMode = 2;
  constexpr int64_t kPadSlotId = -1;
  constexpr int64_t kForwardRunMode = 0;
  const std::optional<Tensor> bias;
  const IntArray no_accepted_tokens;
  auto packed = torch::empty(x.sizes(), x.options().dtype(torch::kBFloat16));
  EXEC_NPU_CMD_EXT(aclnnXllmKernelCausalConv1dQkv,
                   x,
                   weight,
                   bias,
                   conv_state,
                   query_start_loc,
                   state_indices,
                   has_initial_state,
                   no_accepted_tokens,
                   kPackedQkvActivationMode,
                   kPadSlotId,
                   kForwardRunMode,
                   /*q_dim=*/qk_dim,
                   /*k_dim=*/qk_dim,
                   v_dim,
                   head_k_dim,
                   packed);

  const int64_t tokens = x.size(0);
  const auto flat = packed.view({-1});
  const int64_t qk_elements = tokens * qk_dim;
  return {flat.narrow(0, 0, qk_elements)
              .view({1, tokens, num_qk_heads, head_k_dim}),
          flat.narrow(0, qk_elements, qk_elements)
              .view({1, tokens, num_qk_heads, head_k_dim}),
          flat.narrow(0, 2 * qk_elements, tokens * v_dim)
              .view({1, tokens, num_v_heads, head_v_dim})};
}

std::tuple<Tensor, Tensor> mega_chunk_gdn(const Tensor& q,
                                          const Tensor& k,
                                          const Tensor& v,
                                          const Tensor& g,
                                          const Tensor& beta,
                                          const Tensor& mask_lower,
                                          const Tensor& mask_full,
                                          const Tensor& minus_identity,
                                          const Tensor& cu_seqlens,
                                          const Tensor& initial_state,
                                          int64_t num_chunks) {
  CHECK(q.dim() == 4 && q.size(0) == 1);
  const torch::DeviceGuard guard(q.device());
  constexpr int64_t kChunkSize = 128;
  constexpr bool kHasInitialState = true;
  const int64_t sequences = cu_seqlens.numel() - 1;
  const int64_t tokens = q.size(1);
  const int64_t k_dim = q.size(3);
  const int64_t heads = v.size(2);
  const int64_t v_dim = v.size(3);
  const int64_t num_matrices = num_chunks * heads;
  const auto opts = q.options();
  const auto fp32 = opts.dtype(torch::kFloat32);
  auto output = torch::empty({1, tokens, heads, v_dim}, opts);
  auto gate_sum = torch::empty({1, tokens, heads}, fp32);
  auto gate_t = torch::empty({heads, tokens}, fp32);
  auto beta_t = torch::empty({heads, tokens}, opts);
  auto a = torch::zeros({1, tokens, heads, kChunkSize}, opts);
  auto a_inv_f32 = torch::zeros({1, tokens, heads, kChunkSize}, fp32);
  auto a_inv = torch::zeros({1, tokens, heads, kChunkSize}, opts);
  auto w = torch::empty({1, tokens, heads, v_dim}, opts);
  auto u = torch::empty({1, tokens, heads, v_dim}, opts);
  auto h = torch::zeros({num_matrices, k_dim, v_dim}, opts);
  auto v_new = torch::empty_like(v);
  auto final_state = torch::zeros({sequences * heads, k_dim, v_dim}, opts);
  EXEC_NPU_CMD_EXT(aclnnXllmKernelMegaChunkGdn,
                   q,
                   k,
                   v,
                   g,
                   beta,
                   mask_lower,
                   mask_full,
                   minus_identity,
                   cu_seqlens,
                   initial_state,
                   num_matrices,
                   kHasInitialState,
                   output,
                   gate_sum,
                   gate_t,
                   beta_t,
                   a,
                   a_inv_f32,
                   a_inv,
                   w,
                   u,
                   h,
                   v_new,
                   final_state);
  return {output, final_state.view({sequences, heads, k_dim, v_dim})};
}

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("prepare_ascendc_gdn() -> ()", TORCH_FN(prepare_ascendc_gdn));
  m.def(
      "causal_conv1d(Tensor input, Tensor weight, Tensor(a!) conv_state, "
      "int[] query_start_loc, int activation_mode, int run_mode) -> Tensor");
  m.def(
      "causal_conv1d_qkv_prefill(Tensor x, Tensor weight, Tensor(a!) "
      "conv_state, "
      "int[] query_start_loc, int[] state_indices, int[] has_initial_state, "
      "int num_qk_heads, int num_v_heads, int head_k_dim, int head_v_dim) "
      "-> (Tensor, Tensor, Tensor)");
  m.def(
      "mega_chunk_gdn(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
      "Tensor mask_lower, Tensor mask_full, Tensor minus_identity, Tensor "
      "cu_seqlens, "
      "Tensor initial_state, int num_chunks) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("causal_conv1d", TORCH_FN(causal_conv1d));
  m.impl("causal_conv1d_qkv_prefill", TORCH_FN(causal_conv1d_qkv_prefill));
  m.impl("mega_chunk_gdn", TORCH_FN(mega_chunk_gdn));
}

}  // namespace
}  // namespace xllm_kernel
