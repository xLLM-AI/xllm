/* Copyright 2025-2026 The xLLM Authors.

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

#include <glog/logging.h>
#include <tvm/ffi/extra/stl.h>
#include <unistd.h>

#include <cstdlib>
#include <string>
#include <tuple>

#include "kernels/musa/musa_ops_api.h"
#include "kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {

constexpr const char* kGemmOpsUri = "gemm_ops";
constexpr const char* kMusaTopkUri = "sglang_musa_topk_gating";

void check_masked_moe_inputs(const torch::Tensor& input,
                             const torch::Tensor& weights,
                             const torch::Tensor& token_counts,
                             int64_t expected_tokens) {
  TORCH_CHECK(input.defined() && weights.defined() && token_counts.defined(),
              "Mate masked MoE GEMM received an undefined tensor.");
  TORCH_CHECK(input.dim() == 3,
              "Mate masked MoE input must be [experts, tokens, K], got ",
              input.sizes());
  TORCH_CHECK(weights.dim() == 3,
              "Mate masked MoE weights must be [experts, N, K], got ",
              weights.sizes());
  TORCH_CHECK(token_counts.dim() == 1 && token_counts.size(0) == input.size(0),
              "Mate masked MoE token_counts must have one entry per expert.");
  TORCH_CHECK(input.size(0) == weights.size(0) && input.size(1) > 0 &&
                  input.size(2) == weights.size(2),
              "Mate masked MoE input/weight shapes are incompatible: input=",
              input.sizes(),
              " weights=",
              weights.sizes());
  TORCH_CHECK(input.is_contiguous() && weights.is_contiguous() &&
                  token_counts.is_contiguous(),
              "Mate masked MoE tensors must be contiguous.");
  TORCH_CHECK(token_counts.scalar_type() == torch::kInt32,
              "Mate masked MoE token_counts must be int32.");
  TORCH_CHECK(expected_tokens >= 0,
              "Mate masked MoE expected_tokens must be non-negative.");
}

}  // namespace

torch::Tensor masked_moe_gemm_bf16(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& token_counts,
                                   torch::ScalarType output_dtype,
                                   int64_t expected_tokens) {
  check_masked_moe_inputs(input, weights, token_counts, expected_tokens);
  MusaTvmffiStreamGuard stream_guard(input.device());

  auto output = torch::empty({input.size(0), input.size(1), weights.size(1)},
                             input.options().dtype(output_dtype));
  get_function(kGemmOpsUri, "masked_moe_gemm_16bit")(
      to_ffi_tensor_view(input),
      to_ffi_tensor_view(weights),
      to_ffi_tensor_view(token_counts),
      to_ffi_tensor_view(output),
      expected_tokens,
      to_ffi_optional_tensor(std::nullopt));
  return output;
}

torch::Tensor masked_moe_gemm_fp8(const torch::Tensor& input,
                                  const torch::Tensor& input_scale,
                                  const torch::Tensor& weights,
                                  const torch::Tensor& weight_scale,
                                  const torch::Tensor& token_counts,
                                  torch::ScalarType output_dtype,
                                  int64_t expected_tokens) {
  check_masked_moe_inputs(input, weights, token_counts, expected_tokens);
  TORCH_CHECK(input.scalar_type() == torch::kFloat8_e4m3fn &&
                  weights.scalar_type() == torch::kFloat8_e4m3fn,
              "Mate FP8 MoE GEMM currently requires e4m3 inputs and weights.");
  TORCH_CHECK(input_scale.scalar_type() == torch::kFloat32 &&
                  weight_scale.scalar_type() == torch::kFloat32,
              "Mate FP8 MoE scales must be float32.");
  TORCH_CHECK(input_scale.dim() == 3 && weight_scale.dim() == 3,
              "Mate FP8 MoE scales must be 3-D.");
  TORCH_CHECK(input_scale.size(0) == input.size(0) &&
                  input_scale.size(1) == input.size(1) &&
                  input_scale.size(2) * 128 == input.size(2),
              "Mate FP8 input scale shape does not match input.");
  TORCH_CHECK(weight_scale.size(0) == weights.size(0) &&
                  weight_scale.size(1) * 128 == weights.size(1) &&
                  weight_scale.size(2) * 128 == weights.size(2),
              "Mate FP8 weight scale shape does not match weights.");
  TORCH_CHECK(input_scale.is_contiguous() && weight_scale.is_contiguous(),
              "Mate FP8 MoE scales must be contiguous.");

  MusaTvmffiStreamGuard stream_guard(input.device());
  auto output = torch::empty({input.size(0), input.size(1), weights.size(1)},
                             input.options().dtype(output_dtype));
  get_function(kGemmOpsUri, "masked_moe_gemm_8bit")(
      std::make_tuple(to_ffi_borrowed_tensor(input),
                      to_ffi_borrowed_tensor(input_scale)),
      std::make_tuple(to_ffi_borrowed_tensor(weights),
                      to_ffi_borrowed_tensor(weight_scale)),
      to_ffi_tensor_view(token_counts),
      std::make_tuple(static_cast<int64_t>(1),
                      static_cast<int64_t>(128),
                      static_cast<int64_t>(128)),
      to_ffi_tensor_view(output),
      expected_tokens,
      to_ffi_optional_tensor(std::nullopt));
  return output;
}

torch::Tensor contiguous_moe_gemm_bf16(const torch::Tensor& input,
                                       const torch::Tensor& weights,
                                       const torch::Tensor& token_counts,
                                       torch::ScalarType output_dtype) {
  CHECK(input.defined() && weights.defined() && token_counts.defined())
      << "Mate contiguous BF16 MoE GEMM received an undefined tensor.";
  CHECK_EQ(input.dim(), 2)
      << "Mate contiguous BF16 MoE input must be [tokens, K], got "
      << input.sizes();
  CHECK_EQ(weights.dim(), 3)
      << "Mate contiguous BF16 MoE weights must be [experts, N, K], got "
      << weights.sizes();
  CHECK_EQ(token_counts.dim(), 1);
  CHECK_EQ(token_counts.size(0), weights.size(0));
  CHECK_EQ(input.size(1), weights.size(2));
  CHECK(input.is_contiguous() && weights.is_contiguous() &&
        token_counts.is_contiguous())
      << "Mate contiguous BF16 MoE tensors must be contiguous.";
  CHECK_EQ(input.scalar_type(), torch::kBFloat16);
  CHECK_EQ(weights.scalar_type(), torch::kBFloat16);
  CHECK_EQ(token_counts.scalar_type(), torch::kInt32);

  MusaTvmffiStreamGuard stream_guard(input.device());
  torch::Tensor output = torch::empty({input.size(0), weights.size(1)},
                                      input.options().dtype(output_dtype));
  get_function(kGemmOpsUri, "m_grouped_contig_gemm_16bit")(
      to_ffi_tensor_view(input),
      to_ffi_tensor_view(weights),
      to_ffi_tensor_view(token_counts),
      to_ffi_tensor_view(output),
      std::string("K"),
      std::string("K"),
      ffi::Optional<int64_t>());
  return output;
}

torch::Tensor ragged_moe_gemm_bf16(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& row_expert_ids,
                                   torch::ScalarType output_dtype,
                                   int64_t alignment) {
  CHECK(input.defined() && weights.defined() && row_expert_ids.defined())
      << "Mate Ragged BF16 MoE GEMM received an undefined tensor.";
  CHECK_EQ(input.dim(), 2);
  CHECK_EQ(weights.dim(), 3);
  CHECK_EQ(row_expert_ids.dim(), 1);
  CHECK_EQ(row_expert_ids.size(0), input.size(0));
  CHECK_EQ(input.size(1), weights.size(2));
  CHECK(input.is_contiguous() && weights.is_contiguous() &&
        row_expert_ids.is_contiguous())
      << "Mate Ragged BF16 MoE tensors must be contiguous.";
  CHECK_EQ(input.scalar_type(), torch::kBFloat16);
  CHECK_EQ(weights.scalar_type(), torch::kBFloat16);
  CHECK_EQ(row_expert_ids.scalar_type(), torch::kInt32);
  CHECK(alignment == 128 || alignment == 256)
      << "Mate Ragged BF16 MoE alignment must be 128 or 256.";
  CHECK_EQ(input.size(0) % alignment, 0);

  MusaTvmffiStreamGuard stream_guard(input.device());
  torch::Tensor output = torch::empty({input.size(0), weights.size(1)},
                                      input.options().dtype(output_dtype));
  get_function(kGemmOpsUri, "ragged_moe_gemm_16bit")(
      to_ffi_tensor_view(input),
      to_ffi_tensor_view(weights),
      to_ffi_tensor_view(row_expert_ids),
      to_ffi_tensor_view(output),
      /*use_psum_layout=*/false,
      ffi::Optional<int64_t>(),
      alignment);
  return output;
}

torch::Tensor contiguous_moe_gemm_fp8(const torch::Tensor& input,
                                      const torch::Tensor& input_scale,
                                      const torch::Tensor& weights,
                                      const torch::Tensor& weight_scale,
                                      const torch::Tensor& token_counts,
                                      torch::ScalarType output_dtype) {
  CHECK(input.defined() && input_scale.defined() && weights.defined() &&
        weight_scale.defined() && token_counts.defined())
      << "Mate contiguous FP8 MoE GEMM received an undefined tensor.";
  CHECK_EQ(input.dim(), 2)
      << "Mate contiguous FP8 MoE input must be [tokens, K], got "
      << input.sizes();
  CHECK_EQ(weights.dim(), 3)
      << "Mate contiguous FP8 MoE weights must be [experts, N, K], got "
      << weights.sizes();
  CHECK_EQ(token_counts.dim(), 1);
  CHECK_EQ(token_counts.size(0), weights.size(0));
  CHECK_EQ(input.size(1), weights.size(2));
  CHECK(input.is_contiguous() && input_scale.is_contiguous() &&
        weights.is_contiguous() && weight_scale.is_contiguous() &&
        token_counts.is_contiguous())
      << "Mate contiguous FP8 MoE tensors must be contiguous.";
  CHECK_EQ(input.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(weights.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(input_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(weight_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(token_counts.scalar_type(), torch::kInt32);
  CHECK_EQ(input_scale.dim(), 2);
  CHECK_EQ(input_scale.size(0), input.size(0));
  CHECK_EQ(input_scale.size(1) * 128, input.size(1));
  CHECK_EQ(weight_scale.dim(), 3);
  CHECK_EQ(weight_scale.size(0), weights.size(0));
  CHECK_EQ(weight_scale.size(1) * 128, weights.size(1));
  CHECK_EQ(weight_scale.size(2) * 128, weights.size(2));

  MusaTvmffiStreamGuard stream_guard(input.device());
  torch::Tensor output = torch::empty({input.size(0), weights.size(1)},
                                      input.options().dtype(output_dtype));
  get_function(kGemmOpsUri, "m_grouped_contig_gemm_8bit")(
      std::make_tuple(to_ffi_borrowed_tensor(input),
                      to_ffi_borrowed_tensor(input_scale)),
      std::make_tuple(to_ffi_borrowed_tensor(weights),
                      to_ffi_borrowed_tensor(weight_scale)),
      to_ffi_tensor_view(token_counts),
      std::make_tuple(static_cast<int64_t>(1),
                      static_cast<int64_t>(128),
                      static_cast<int64_t>(128)),
      to_ffi_tensor_view(output),
      std::string("K"),
      std::string("K"),
      to_ffi_optional_tensor(std::nullopt));
  return output;
}

torch::Tensor ragged_moe_gemm_fp8(const torch::Tensor& input,
                                  const torch::Tensor& input_scale,
                                  const torch::Tensor& weights,
                                  const torch::Tensor& weight_scale,
                                  const torch::Tensor& row_expert_ids,
                                  torch::ScalarType output_dtype,
                                  int64_t alignment) {
  CHECK(input.defined() && input_scale.defined() && weights.defined() &&
        weight_scale.defined() && row_expert_ids.defined())
      << "Mate Ragged FP8 MoE GEMM received an undefined tensor.";
  CHECK_EQ(input.dim(), 2);
  CHECK_EQ(weights.dim(), 3);
  CHECK_EQ(row_expert_ids.dim(), 1);
  CHECK_EQ(row_expert_ids.size(0), input.size(0));
  CHECK_EQ(input.size(1), weights.size(2));
  CHECK(input.is_contiguous() && input_scale.is_contiguous() &&
        weights.is_contiguous() && weight_scale.is_contiguous() &&
        row_expert_ids.is_contiguous())
      << "Mate Ragged FP8 MoE tensors must be contiguous.";
  CHECK_EQ(input.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(weights.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(input_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(weight_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(row_expert_ids.scalar_type(), torch::kInt32);
  CHECK_EQ(input_scale.dim(), 2);
  CHECK_EQ(input_scale.size(0), input.size(0));
  CHECK_EQ(input_scale.size(1) * 128, input.size(1));
  CHECK_EQ(weight_scale.dim(), 3);
  CHECK_EQ(weight_scale.size(0), weights.size(0));
  CHECK_EQ(weight_scale.size(1) * 128, weights.size(1));
  CHECK_EQ(weight_scale.size(2) * 128, weights.size(2));
  CHECK(alignment == 128 || alignment == 256)
      << "Mate Ragged FP8 MoE alignment must be 128 or 256.";
  CHECK_EQ(input.size(0) % alignment, 0);

  MusaTvmffiStreamGuard stream_guard(input.device());
  torch::Tensor output = torch::empty({input.size(0), weights.size(1)},
                                      input.options().dtype(output_dtype));
  get_function(kGemmOpsUri, "ragged_moe_gemm_8bit")(
      std::make_tuple(to_ffi_borrowed_tensor(input),
                      to_ffi_borrowed_tensor(input_scale)),
      std::make_tuple(to_ffi_borrowed_tensor(weights),
                      to_ffi_borrowed_tensor(weight_scale)),
      to_ffi_tensor_view(row_expert_ids),
      std::make_tuple(static_cast<int64_t>(1),
                      static_cast<int64_t>(128),
                      static_cast<int64_t>(128)),
      to_ffi_tensor_view(output),
      alignment);
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> musa_moe_topk_softmax(
    const torch::Tensor& router_logits,
    int64_t topk) {
  CHECK(router_logits.defined());
  CHECK_EQ(router_logits.dim(), 2);
  CHECK(router_logits.is_contiguous());
  CHECK_EQ(router_logits.scalar_type(), torch::kBFloat16);
  CHECK_GT(topk, 0);
  CHECK_LE(topk, router_logits.size(1));

  MusaTvmffiStreamGuard stream_guard(router_logits.device());
  auto topk_weights =
      torch::empty({router_logits.size(0), topk},
                   router_logits.options().dtype(torch::kFloat32));
  auto topk_ids = torch::empty({router_logits.size(0), topk},
                               router_logits.options().dtype(torch::kInt32));
  auto unused_correction_bias = topk_weights.reshape({-1});
  get_function(kMusaTopkUri, "sgl_musa_topk_softmax")(
      to_ffi_tensor_view(topk_weights),
      to_ffi_tensor_view(topk_ids),
      to_ffi_tensor_view(router_logits),
      /*renormalize=*/true,
      /*moe_softcapping=*/0.0,
      to_ffi_tensor_view(unused_correction_bias),
      /*has_correction_bias=*/false);
  return std::make_tuple(topk_weights, topk_ids);
}

bool musa_moe_topk_softmax_available() {
  static const bool available = [] {
    const char* ops_path = std::getenv("FLASHINFER_OPS_PATH");
    if (ops_path == nullptr || ops_path[0] == '\0') {
      return false;
    }
    const std::string so_path =
        std::string(ops_path) + "/" + kMusaTopkUri + "/" + kMusaTopkUri + ".so";
    return ::access(so_path.c_str(), R_OK) == 0;
  }();
  return available;
}

}  // namespace xllm::kernel::musa
