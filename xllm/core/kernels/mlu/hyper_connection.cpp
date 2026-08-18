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

#include "mlu_ops_api.h"

namespace xllm::kernel::mlu {

torch::Tensor fused_mul_reduce_sum(const torch::Tensor& x,
                                   const torch::Tensor& w) {
  return tmo::torch_api::fused_mul_reduce_sum(x, w);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> hc_split_sinkhorn(
    const torch::Tensor& mixes,
    const torch::Tensor& hc_scale,
    const torch::Tensor& hc_base,
    const std::optional<torch::Tensor>& pre_scale,
    int64_t hc_mult,
    int64_t sinkhorn_iter,
    double eps) {
  std::vector<torch::Tensor> outputs = tmo::torch_api::hc_split_sinkhorn(
      mixes, hc_scale, hc_base, pre_scale, hc_mult, sinkhorn_iter, eps);
  return {outputs[0], outputs[1], outputs[2]};
}

std::tuple<torch::Tensor, torch::Tensor> fused_mhc_post(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb,
    bool compute_rms,
    double eps) {
  torch::Tensor output = torch::empty_like(residual);
  torch::Tensor output_rms;
  if (compute_rms) {
    output_rms = torch::empty({x.size(0)}, x.options().dtype(torch::kFloat32));
  } else {
    output_rms = torch::empty({0}, x.options().dtype(torch::kFloat32));
  }
  tmo::torch_api::fused_mhc_post(
      x, residual, post, comb, output, output_rms, compute_rms, eps);
  if (!compute_rms) {
    output_rms = torch::Tensor();
  }
  return {output, output_rms};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_mhc(const torch::Tensor& x,
          const torch::Tensor& residual_in,
          const torch::Tensor& hc_fn,
          const torch::Tensor& gamma,
          const torch::Tensor& post_in,
          const torch::Tensor& comb_in,
          const torch::Tensor& hc_scale,
          const torch::Tensor& hc_base,
          int64_t sinkhorn_iter,
          double eps) {
  torch::Tensor norm_out = torch::empty_like(x);
  torch::Tensor residual_out = torch::empty_like(residual_in);
  torch::Tensor post_out = torch::empty_like(post_in);
  torch::Tensor comb_out = torch::empty_like(comb_in);
  tmo::torch_api::fused_mhc(x,
                            residual_in,
                            hc_fn,
                            gamma,
                            post_in,
                            comb_in,
                            hc_scale,
                            hc_base,
                            norm_out,
                            residual_out,
                            post_out,
                            comb_out,
                            sinkhorn_iter,
                            eps);
  return {norm_out, residual_out, post_out, comb_out};
}

}  // namespace xllm::kernel::mlu
