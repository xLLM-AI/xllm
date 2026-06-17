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

#include <torch/library.h>

#include <tuple>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

constexpr int64_t kQOutDimCacheMode0 = 576;
constexpr int64_t kQOutDimOtherCacheMode = 512;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
construct_mla_preprocess_output_tensor(const at::Tensor& input,
                                       const at::Tensor& wuk,
                                       const at::Tensor& kv_cache,
                                       const at::Tensor& kv_cache_rope,
                                       int64_t cache_mode,
                                       int64_t q_rope_dim) {
  TORCH_CHECK(input.dim() >= 1,
              "input's dim num should be at least 1, actual ",
              input.dim(),
              ".");
  TORCH_CHECK(wuk.dim() >= 1,
              "wuk's dim num should be at least 1, actual ",
              wuk.dim(),
              ".");

  int64_t token_num = input.size(0);
  int64_t head_num = wuk.size(0);
  int64_t q_out_dim =
      (cache_mode == 0) ? kQOutDimCacheMode0 : kQOutDimOtherCacheMode;

  at::Tensor q_out = at::empty({token_num, head_num, q_out_dim},
                               kv_cache.options().dtype(kv_cache.dtype()));
  at::Tensor kv_cache_out =
      at::empty(kv_cache.sizes(), kv_cache.options().dtype(kv_cache.dtype()));
  at::Tensor q_rope_out = at::empty({token_num, head_num, q_rope_dim},
                                    input.options().dtype(input.dtype()));
  at::Tensor kr_cache_out =
      at::empty(kv_cache_rope.sizes(), input.options().dtype(input.dtype()));
  return std::make_tuple(q_out, kv_cache_out, q_rope_out, kr_cache_out);
}

void check_mla_preprocess_shape_and_dtype(const at::Tensor& input,
                                          const at::Tensor& wuk,
                                          int64_t q_rope_dim,
                                          int64_t k_rope_dim,
                                          int64_t cache_mode,
                                          int64_t quant_mode,
                                          int64_t wdkv_split_count) {
  TORCH_CHECK(input.dtype() == at::kHalf || input.dtype() == at::kBFloat16,
              "input should be FLOAT16 or BFLOAT16.");
  TORCH_CHECK(wuk.dim() >= 1,
              "wuk's dim num should be at least 1, actual ",
              wuk.dim(),
              ".");
  TORCH_CHECK(q_rope_dim > 0,
              "q_rope_dim should be greater than 0, actual ",
              q_rope_dim,
              ".");
  TORCH_CHECK(k_rope_dim > 0,
              "k_rope_dim should be greater than 0, actual ",
              k_rope_dim,
              ".");
  TORCH_CHECK(cache_mode >= 0,
              "cache_mode should be non-negative, actual ",
              cache_mode,
              ".");
  TORCH_CHECK(quant_mode >= 0,
              "quant_mode should be non-negative, actual ",
              quant_mode,
              ".");
  TORCH_CHECK(wdkv_split_count > 0,
              "wdkv_split_count should be greater than 0, actual ",
              wdkv_split_count,
              ".");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mla_preprocess(
    const at::Tensor& input,
    const at::Tensor& gamma0,
    const at::Tensor& beta0,
    const at::Tensor& quant_scale0,
    const at::Tensor& quant_offset0,
    const at::Tensor& wdqkv,
    const at::Tensor& descale0,
    const at::Tensor& bias0,
    const at::Tensor& gamma1,
    const at::Tensor& beta1,
    const at::Tensor& quant_scale1,
    const at::Tensor& quant_offset1,
    const at::Tensor& wuq,
    const at::Tensor& descale1,
    const at::Tensor& bias1,
    const at::Tensor& gamma2,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& wuk,
    const at::Tensor& kv_cache,
    const at::Tensor& kv_cache_rope,
    const at::Tensor& slot_mapping,
    const at::Tensor& ctkv_scale,
    const at::Tensor& q_nope_scale,
    int64_t wdq_dim,
    int64_t q_rope_dim,
    int64_t k_rope_dim,
    double epsilon,
    int64_t q_rotary_coeff,
    int64_t k_rotary_coeff,
    bool transepose_wdq,
    bool transepose_wuq,
    bool transepose_wuk,
    int64_t cache_mode,
    int64_t quant_mode,
    bool do_rms_norm,
    int64_t wdkv_split_count) {
  check_mla_preprocess_shape_and_dtype(input,
                                       wuk,
                                       q_rope_dim,
                                       k_rope_dim,
                                       cache_mode,
                                       quant_mode,
                                       wdkv_split_count);

  auto output_tensors = construct_mla_preprocess_output_tensor(
      input, wuk, kv_cache, kv_cache_rope, cache_mode, q_rope_dim);
  at::Tensor q_out = std::get<0>(output_tensors);
  at::Tensor kv_cache_out = std::get<1>(output_tensors);
  at::Tensor q_rope_out = std::get<2>(output_tensors);
  at::Tensor kr_cache_out = std::get<3>(output_tensors);

  EXEC_NPU_CMD(aclnnMlaPreprocess,
               input,
               gamma0,
               beta0,
               quant_scale0,
               quant_offset0,
               wdqkv,
               descale0,
               bias0,
               gamma1,
               beta1,
               quant_scale1,
               quant_offset1,
               wuq,
               descale1,
               bias1,
               gamma2,
               cos,
               sin,
               wuk,
               kv_cache,
               kv_cache_rope,
               slot_mapping,
               ctkv_scale,
               q_nope_scale,
               wdq_dim,
               q_rope_dim,
               k_rope_dim,
               epsilon,
               q_rotary_coeff,
               k_rotary_coeff,
               transepose_wdq,
               transepose_wuq,
               transepose_wuk,
               cache_mode,
               quant_mode,
               do_rms_norm,
               wdkv_split_count,
               q_out,
               kv_cache_out,
               q_rope_out,
               kr_cache_out);

  return std::make_tuple(q_out, kv_cache_out, q_rope_out, kr_cache_out);
}

}  // namespace xllm::kernel::npu
