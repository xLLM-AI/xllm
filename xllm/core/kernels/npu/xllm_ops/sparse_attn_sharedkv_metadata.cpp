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

#include <cstdlib>
#include <iostream>

#include <torch/library.h>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

namespace {

auto get_valid_tensor = [](const c10::optional<at::Tensor>& tensor_opt,
                           at::Device device) {
  return tensor_opt.has_value()
             ? tensor_opt
	             : torch::empty({0}, torch::dtype(torch::kInt32).device(device));
};

void dump_optional_tensor(const char* name,
                          const c10::optional<at::Tensor>& tensor_opt) {
  std::cerr << " " << name << "=";
  if (!tensor_opt.has_value()) {
    std::cerr << "None";
    return;
  }
  const auto& tensor = tensor_opt.value();
  std::cerr << "defined=" << tensor.defined();
  if (!tensor.defined()) {
    return;
  }
  std::cerr << " sizes=" << tensor.sizes() << " dtype=" << tensor.scalar_type()
            << " device=" << tensor.device() << " numel=" << tensor.numel();
  if (tensor.numel() > 0 && tensor.numel() <= 8 &&
      tensor.scalar_type() == at::kInt) {
    auto cpu = tensor.device().is_cpu() ? tensor.contiguous()
                                       : tensor.cpu().contiguous();
    std::cerr << " vals=[";
    auto* data = cpu.data_ptr<int32_t>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
      if (i > 0) {
        std::cerr << ",";
      }
      std::cerr << data[i];
    }
    std::cerr << "]";
  }
}

}  // namespace

at::Tensor sparse_attn_sharedkv_metadata(
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim,
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_ori_kv,
    const c10::optional<at::Tensor>& cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_kv,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    int64_t ori_topk,
    int64_t cmp_topk,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    bool has_ori_kv,
    bool has_cmp_kv) {
  at::Device output_device(std::string("npu"));
  if (cu_seqlens_q.has_value()) {
    output_device = cu_seqlens_q.value().device();
  } else if (cu_seqlens_ori_kv.has_value()) {
    output_device = cu_seqlens_ori_kv.value().device();
  } else if (cu_seqlens_cmp_kv.has_value()) {
    output_device = cu_seqlens_cmp_kv.value().device();
  } else if (seqused_q.has_value()) {
    output_device = seqused_q.value().device();
  } else if (seqused_kv.has_value()) {
    output_device = seqused_kv.value().device();
  }

  at::Tensor output =
      torch::zeros({kDsaMetadataBufferElements},
                   torch::dtype(torch::kInt32).device(output_device));
  auto cu_seqlens_q_val = get_valid_tensor(cu_seqlens_q, output_device);
  auto cu_seqlens_ori_kv_val =
      get_valid_tensor(cu_seqlens_ori_kv, output_device);
  auto cu_seqlens_cmp_kv_val =
      get_valid_tensor(cu_seqlens_cmp_kv, output_device);
  auto seqused_q_val = get_valid_tensor(seqused_q, output_device);
  auto seqused_kv_val = get_valid_tensor(seqused_kv, output_device);

  if (std::getenv("XLLM_DSV4_SPMETA_CPP_DEBUG") != nullptr) {
    std::cerr << "[SPMETA CPP]"
              << " heads_q=" << num_heads_q << " heads_kv=" << num_heads_kv
              << " head_dim=" << head_dim << " batch=" << batch_size
              << " max_q=" << max_seqlen_q << " max_kv=" << max_seqlen_kv
              << " ori_topk=" << ori_topk << " cmp_topk=" << cmp_topk
              << " cmp_ratio=" << cmp_ratio
              << " ori_mask=" << ori_mask_mode
              << " cmp_mask=" << cmp_mask_mode
              << " ori_win_left=" << ori_win_left
              << " ori_win_right=" << ori_win_right
              << " layout_q=" << layout_q << " layout_kv=" << layout_kv
              << " has_ori=" << has_ori_kv << " has_cmp=" << has_cmp_kv
              << " out_device=" << output_device;
    dump_optional_tensor("cu_q", cu_seqlens_q);
    dump_optional_tensor("cu_ori", cu_seqlens_ori_kv);
    dump_optional_tensor("cu_cmp", cu_seqlens_cmp_kv);
    dump_optional_tensor("seqused_q", seqused_q);
    dump_optional_tensor("seqused_kv", seqused_kv);
    std::cerr << std::endl;
  }

  // convert str
  std::string layout_q_str = std::string(layout_q);
  std::string layout_kv_str = std::string(layout_kv);
  char* layout_q_ptr = const_cast<char*>(layout_q_str.c_str());
  char* layout_kv_ptr = const_cast<char*>(layout_kv_str.c_str());

  EXEC_NPU_CMD(aclnnSparseAttnSharedkvMetadata,
               cu_seqlens_q_val,
               cu_seqlens_ori_kv_val,
               cu_seqlens_cmp_kv_val,
               seqused_q_val,
               seqused_kv_val,
               num_heads_q,
               num_heads_kv,
               head_dim,
               batch_size,
               max_seqlen_q,
               max_seqlen_kv,
               ori_topk,
               cmp_topk,
               cmp_ratio,
               ori_mask_mode,
               cmp_mask_mode,
               ori_win_left,
               ori_win_right,
               layout_q_ptr,
               layout_kv_ptr,
               has_ori_kv,
               has_cmp_kv,
               output);

  return output;
}

}  // namespace xllm::kernel::npu
