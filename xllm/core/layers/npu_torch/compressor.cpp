/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "compressor.h"

#include <glog/logging.h>

#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#if defined(USE_NPU)
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#endif
#endif

#include "kernels/ops_api.h"

DECLARE_bool(enable_chunked_prefill);
namespace xllm {
namespace layer {

namespace {

int64_t tensor_bytes(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return 0;
  }
  return tensor.numel() * tensor.element_size();
}

torch::Tensor get_first_defined_tensor(const StateDict& state_dict,
                                       const std::vector<std::string>& names) {
  for (const auto& name : names) {
    auto tensor = state_dict.get_tensor(name);
    if (tensor.defined()) {
      return tensor;
    }
  }
  return torch::Tensor();
}

std::string list_available_keys(const StateDict& state_dict) {
  std::ostringstream os;
  int count = 0;
  for (const auto& kv : state_dict) {
    if (count++ > 0) {
      os << ", ";
    }
    os << kv.first;
    if (count >= 20) {
      os << ", ...";
      break;
    }
  }
  return os.str();
}

}  // namespace

CompressorImpl::CompressorImpl(int64_t compress_ratio, int64_t head_dim)
    : CompressorImpl(
          compress_ratio,
          head_dim,
          64,
          2,
          1e-6,
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)) {}

CompressorImpl::CompressorImpl(int64_t compress_ratio,
                               int64_t head_dim,
                               const torch::TensorOptions& options)
    : CompressorImpl(compress_ratio, head_dim, 64, 2, 1e-6, options) {}

CompressorImpl::CompressorImpl(int64_t compress_ratio,
                               int64_t head_dim,
                               int64_t rope_head_dim,
                               int64_t rot_mode,
                               double norm_eps,
                               const torch::TensorOptions& options)
    : compress_ratio_(compress_ratio),
      head_dim_(head_dim),
      rope_head_dim_(rope_head_dim),
      rot_mode_(rot_mode),
      eps_(norm_eps),
      options_(options) {
  enable_compressor_overlap_ = (compress_ratio == 4);
}

torch::Tensor CompressorImpl::forward(
    const DSAMetadata& attn_metadata,
    torch::Tensor& hidden_states,
    std::tuple<torch::Tensor, torch::Tensor>& kv_states,
    std::tuple<torch::Tensor, torch::Tensor>& block_tables,
    torch::Tensor& compressed_sin,
    torch::Tensor& compressed_cos,
    torch::Tensor actual_seq_lengths_query) {
  CHECK(cmp_wkv_loaded_ && cmp_wgate_loaded_ && cmp_norm_loaded_ &&
        cmp_ape_loaded_)
      << "Compressor weights are incomplete before forward, prefix="
      << (last_load_prefix_.empty() ? "<unknown>" : last_load_prefix_)
      << ", loaded={wkv:" << (cmp_wkv_loaded_ ? "true" : "false")
      << ", wgate:" << (cmp_wgate_loaded_ ? "true" : "false")
      << ", norm:" << (cmp_norm_loaded_ ? "true" : "false")
      << ", ape:" << (cmp_ape_loaded_ ? "true" : "false")
      << "}. This usually means weights were not found across state_dict "
         "shards.";

  auto [kv_state, score_state] = kv_states;
  auto [kv_block_table, score_block_table] = block_tables;

  const int64_t sin_last_dim = compressed_sin.size(compressed_sin.dim() - 1);
  const int64_t cos_last_dim = compressed_cos.size(compressed_cos.dim() - 1);

  torch::Tensor compressed_kv;
  // TODO - replace opfunc; cu_seqlens/start_pos need Tensor from DSA metadata
  xllm::kernel::CompressorParams params;
  params.x = hidden_states;
  params.wkv = cmp_wkv_;
  params.wgate = cmp_wgate_;
  params.kv_state = kv_state;
  params.score_state = score_state;
  params.ape = cmp_ape_;
  params.norm_weight = cmp_norm_;
  params.rope_sin = compressed_sin.view({-1, sin_last_dim});
  params.rope_cos = compressed_cos.view({-1, cos_last_dim});
  params.kv_block_table = c10::optional<torch::Tensor>(kv_block_table);
  params.score_block_table = c10::optional<torch::Tensor>(score_block_table);
  params.cu_seqlens = c10::optional<torch::Tensor>(actual_seq_lengths_query);
  params.seqused = c10::nullopt;
  params.start_pos = c10::optional<torch::Tensor>(attn_metadata.start_pos);
  params.rope_head_dim = rope_head_dim_;
  params.cmp_ratio = compress_ratio_;
  params.coff = enable_compressor_overlap_ ? 2 : 1;
  params.norm_eps = eps_;
  params.rotary_mode = rot_mode_;
  params.enable_grad = false;

  std::tie(compressed_kv, std::ignore, std::ignore, std::ignore, std::ignore) =
      xllm::kernel::compressor(params);
  return compressed_kv;
}

void CompressorImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  last_load_prefix_ = std::string(state_dict.prefix());

  auto wkv = get_first_defined_tensor(state_dict,
                                      {"wkv.weight",
                                       "wkv",
                                       "wkv.linear.weight",
                                       "wkv_proj.weight",
                                       "kv_proj.weight"});
  if (!wkv.defined()) {
    // Compatibility fallback: some checkpoints split compressor projection
    // into wk / wv and can be merged along output dim.
    const auto wk = get_first_defined_tensor(
        state_dict, {"wk.weight", "wk", "k_proj.weight"});
    const auto wv = get_first_defined_tensor(
        state_dict, {"wv.weight", "wv", "v_proj.weight"});
    if (wk.defined() && wv.defined()) {
      wkv = torch::cat({wk, wv}, /*dim=*/0);
    }
  }

  const auto wgate = get_first_defined_tensor(state_dict,
                                              {"wgate.weight",
                                               "wgate",
                                               "wgate.linear.weight",
                                               "gate.weight",
                                               "score_proj.weight"});
  const auto norm = get_first_defined_tensor(
      state_dict,
      {"norm.weight", "norm", "layer_norm.weight", "rms_norm.weight"});
  auto ape = get_first_defined_tensor(
      state_dict, {"ape", "ape.weight", "position_bias", "rope_bias"});

  if (!wkv.defined() && !wgate.defined() && !norm.defined() && !ape.defined()) {
    // In multi-file checkpoints each call may contain only part of weights.
    return;
  }

  auto coff = enable_compressor_overlap_ ? 2 : 1;
  const bool is_npu_device =
      options_.device().type() == c10::DeviceType::PrivateUse1;

  if (wkv.defined()) {
    cmp_wkv_ = wkv.to(options_);
    cmp_wkv_loaded_ = true;
  }
  if (wgate.defined()) {
    cmp_wgate_ = wgate.to(options_);
    cmp_wgate_loaded_ = true;
  }
  if (norm.defined()) {
    cmp_norm_ = norm.to(options_);
    cmp_norm_loaded_ = true;
  }
  if (ape.defined()) {
    CHECK_EQ(ape.dim(), 2) << "ape weight dim should be 2, but got "
                           << ape.dim();
    CHECK_EQ(ape.size(0), compress_ratio_)
        << "ape weight size mismatch on dim0, expected " << compress_ratio_
        << " but got " << ape.size(0);
    CHECK_EQ(ape.size(1), coff * head_dim_)
        << "ape weight size mismatch on dim1, expected " << (coff * head_dim_)
        << " but got " << ape.size(1);
    cmp_ape_ = ape.to(options_.dtype(torch::kFloat32));
    cmp_ape_loaded_ = true;
  }

  if (cmp_wkv_loaded_) {
    CHECK_EQ(cmp_wkv_.device(), options_.device())
        << "cmp_wkv device mismatch, expected: " << options_.device()
        << ", actual: " << cmp_wkv_.device();
  }
  if (cmp_wgate_loaded_) {
    CHECK_EQ(cmp_wgate_.device(), options_.device())
        << "cmp_wgate device mismatch, expected: " << options_.device()
        << ", actual: " << cmp_wgate_.device();
  }
  if (cmp_norm_loaded_) {
    CHECK_EQ(cmp_norm_.device(), options_.device())
        << "cmp_norm device mismatch, expected: " << options_.device()
        << ", actual: " << cmp_norm_.device();
  }
  if (cmp_ape_loaded_) {
    CHECK_EQ(cmp_ape_.device(), options_.device())
        << "cmp_ape device mismatch, expected: " << options_.device()
        << ", actual: " << cmp_ape_.device();
  }
}

int64_t CompressorImpl::weight_bytes() const {
  return tensor_bytes(cmp_wkv_) + tensor_bytes(cmp_wgate_) +
         tensor_bytes(cmp_norm_) + tensor_bytes(cmp_ape_);
}

}  // namespace layer
}  // namespace xllm
