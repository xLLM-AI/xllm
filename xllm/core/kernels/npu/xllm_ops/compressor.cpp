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
#include <torch/library.h>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

namespace {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
construct_compressor_output_tensor(const at::Tensor& x,
                                   const at::Tensor& norm_weight,
                                   const at::Tensor& rope_sin,
                                   int64_t cmp_ratio,
                                   int64_t coff,
                                   bool enable_grad) {
  constexpr int32_t DIM_1 = 1;
  constexpr int32_t DIM_2 = 2;
  constexpr int32_t DIM_3 = 3;
  constexpr int32_t VALUE_0 = 0;
  int64_t x_dim = x.dim();
  at::SmallVector<int64_t, 8> cmp_kv_size;
  at::SmallVector<int64_t, 8> wkv_proj_size;
  at::SmallVector<int64_t, 8> softmax_res_size;
  at::SmallVector<int64_t, 8> norm_x_size;
  at::SmallVector<int64_t, 8> norm_rstd_size;
  at::Tensor cmp_kv;
  at::Tensor wkv_proj;
  at::Tensor softmax_res;
  at::Tensor norm_x;
  at::Tensor norm_rstd;
  int64_t cmp_s = 0;
  if (x_dim == DIM_3) {
    cmp_s = (x.size(1) + cmp_ratio - 1) / cmp_ratio;
    cmp_kv_size = {x.size(0), cmp_s, norm_weight.size(0)};
    if (enable_grad) {
      wkv_proj_size = {x.size(0), x.size(1), coff * norm_weight.size(0)};
      softmax_res_size = {
          x.size(0), cmp_s, coff * cmp_ratio, norm_weight.size(0)};
      norm_x_size = {x.size(0), cmp_s, norm_weight.size(0)};
      norm_rstd_size = {x.size(0), cmp_s};
    }
  } else {
    cmp_s = rope_sin.size(0);
    cmp_kv_size = {cmp_s, norm_weight.size(0)};
    if (enable_grad) {
      wkv_proj_size = {x.size(0), coff * norm_weight.size(0)};
      softmax_res_size = {cmp_s, coff * cmp_ratio, norm_weight.size(0)};
      norm_x_size = {cmp_s, norm_weight.size(0)};
      norm_rstd_size = {cmp_s};
    }
  }

  cmp_kv = at::zeros(cmp_kv_size, x.options().dtype(x.dtype()));
  if (enable_grad) {
    wkv_proj = at::zeros(wkv_proj_size, x.options().dtype(x.dtype()));
    softmax_res = at::zeros(softmax_res_size, x.options().dtype(x.dtype()));
    norm_x = at::zeros(norm_x_size, x.options().dtype(x.dtype()));
    norm_rstd = at::zeros(norm_rstd_size, x.options().dtype(x.dtype()));
  } else {
    wkv_proj = at::empty({0}, x.options().dtype(x.dtype()));
    softmax_res = at::empty({0}, x.options().dtype(x.dtype()));
    norm_x = at::empty({0}, x.options().dtype(x.dtype()));
    norm_rstd = at::empty({0}, x.options().dtype(x.dtype()));
  }

  return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
      cmp_kv, wkv_proj, softmax_res, norm_x, norm_rstd);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
compressor(const at::Tensor& x,
           const at::Tensor& wkv,
           const at::Tensor& wgate,
           at::Tensor& kv_state,
           at::Tensor& score_state,
           const at::Tensor& ape,
           const at::Tensor& norm_weight,
           const at::Tensor& rope_sin,
           const at::Tensor& rope_cos,
           const c10::optional<at::Tensor>& kv_block_table,
           const c10::optional<at::Tensor>& score_block_table,
           const c10::optional<at::Tensor>& cu_seqlens,
           const c10::optional<at::Tensor>& seqused,
           const c10::optional<at::Tensor>& start_pos,
           int64_t rope_head_dim,
           int64_t cmp_ratio,
           int64_t coff,
           double norm_eps,
           int64_t rotary_mode,
           bool enable_grad) {
  constexpr int32_t DIM_1 = 1;
  constexpr int32_t DIM_2 = 2;
  constexpr int32_t DIM_3 = 3;
  constexpr int32_t VALUE_0 = 0;
  // construct the output tensor
  int64_t x_dim = x.dim();
  CHECK(x_dim == DIM_2 || x_dim == DIM_3)
      << "x dim num[" << x_dim << "] should be 2 or 3";

  int64_t norm_weight_dim = norm_weight.dim();
  CHECK(norm_weight_dim == DIM_1)
      << "norm_weight dim num[" << norm_weight_dim << "] should be 1";

  int64_t rope_sin_dim = rope_sin.dim();
  CHECK(rope_sin_dim == x_dim)
      << "rope_sin dim num[" << rope_sin_dim
      << "] should be equal to x dim num[" << x_dim << "]";

  CHECK(cmp_ratio != VALUE_0) << "cmp_ratio should not be 0";

  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
      output = construct_compressor_output_tensor(
          x, norm_weight, rope_sin, cmp_ratio, coff, enable_grad);
  at::Tensor cmp_kv = std::get<0>(output);
  at::Tensor wkv_proj = std::get<1>(output);
  at::Tensor softmax_res = std::get<2>(output);
  at::Tensor norm_x = std::get<3>(output);
  at::Tensor norm_rstd = std::get<4>(output);

  {
    static thread_local bool k_logged = false;
    if (!k_logged) {
      k_logged = true;
      auto shp = [](const at::Tensor& t) {
        return t.defined() ? t.sizes() : c10::IntArrayRef{};
      };
      auto oshp = [](const c10::optional<at::Tensor>& t) {
        return (t.has_value() && t->defined()) ? t->sizes()
                                               : c10::IntArrayRef{};
      };
      LOG(ERROR) << "[CMP-KERNEL] x=" << shp(x) << " wkv=" << shp(wkv)
                 << " kv_state=" << shp(kv_state)
                 << " score_state=" << shp(score_state)
                 << " rope_sin=" << shp(rope_sin)
                 << " kv_block_table=" << oshp(kv_block_table)
                 << " score_block_table=" << oshp(score_block_table)
                 << " cu_seqlens=" << oshp(cu_seqlens)
                 << " seqused=" << oshp(seqused)
                 << " start_pos=" << oshp(start_pos)
                 << " rope_head_dim=" << rope_head_dim
                 << " cmp_ratio=" << cmp_ratio << " coff=" << coff;
      auto dumpv = [](const char* nm, const c10::optional<at::Tensor>& t) {
        if (!t.has_value() || !t->defined() || t->numel() == 0) {
          LOG(ERROR) << "[CMP-KERNEL]   " << nm << "=<empty>";
          return;
        }
        auto c = t->to(torch::kCPU).to(torch::kLong).contiguous();
        std::string v;
        const int64_t n = std::min<int64_t>(c.numel(), 70);
        for (int64_t i = 0; i < n; ++i)
          v += std::to_string(c.data_ptr<int64_t>()[i]) + ",";
        LOG(ERROR) << "[CMP-KERNEL]   " << nm << "(numel=" << c.numel() << ")=["
                   << v << "]";
      };
      dumpv("kv_block_table", kv_block_table);
      dumpv("cu_seqlens", cu_seqlens);
      dumpv("start_pos", start_pos);
    }
  }
  EXEC_NPU_CMD(aclnnCompressor,
               x,
               wkv,
               wgate,
               kv_state,
               score_state,
               ape,
               norm_weight,
               rope_sin,
               rope_cos,
               kv_block_table,
               score_block_table,
               cu_seqlens,
               seqused,
               start_pos,
               rope_head_dim,
               cmp_ratio,
               coff,
               norm_eps,
               rotary_mode,
               enable_grad,
               cmp_kv,
               wkv_proj,
               softmax_res,
               norm_x,
               norm_rstd);

  return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
      cmp_kv, wkv_proj, softmax_res, norm_x, norm_rstd);
}

}  // namespace xllm::kernel::npu
