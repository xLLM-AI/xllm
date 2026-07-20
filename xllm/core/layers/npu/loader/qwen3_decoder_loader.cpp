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

#include "qwen3_decoder_loader.h"

#include "core/framework/config/kernel_config.h"
#include "core/framework/config/parallel_config.h"
#include "qwen_loader_constants.h"

namespace xllm {
namespace layer {

using namespace qwen3_decoder_constants;

Qwen3DecoderLoader::Qwen3DecoderLoader(uint64_t weight_count,
                                       const ModelContext& context,
                                       bool enableAddNorm,
                                       LoadMode mode)
    : BaseLoader(weight_count, context, mode), enableAddNorm_(enableAddNorm) {
  auto options = context.get_tensor_options();
  rank_id_ = parallel_args_.rank();

  dtype_ = torch::typeMetaToScalarType(options.dtype());
  auto& t = working_tensors();
  auto target_options = options.device(target_device());
  for (uint64_t i = 0; i < weight_count; ++i) {
    t[i] = torch::zeros({1}, target_options);
  }
  at_placeholder_ = torch::zeros({1}).to(target_device()).to(dtype_);
}

void Qwen3DecoderLoader::load_state_dict(const StateDict& state_dict) {
  const bool to_host = load_to_host();
  auto& t = working_tensors();
  if (quantize_type_.compare("w8a8") == 0) {
    for (const auto& [index, name] : WEIGHT_MAPPING_W8A8) {
      if (WEIGHT_SHARD_W8A8.find(index) != WEIGHT_SHARD_W8A8.end()) {
        set_weight(state_dict, name, index, WEIGHT_SHARD_W8A8[index], to_host);
      } else {
        set_weight(state_dict, name, index, to_host);
      }
    }
    t[IN_NORM_BIAS] =
        torch::zeros(t[IN_NORM_WEIGHT].sizes(), t[IN_NORM_WEIGHT].options());
    t[IN_SELFOUT_NORM_BIAS] = torch::zeros(t[IN_SELFOUT_NORM_WEIGHT].sizes(),
                                           t[IN_SELFOUT_NORM_WEIGHT].options());
    return;
  }

  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index], to_host);
    } else {
      set_weight(state_dict, name, index, to_host);
    }
  }
}

void Qwen3DecoderLoader::verify_loaded_weights() const {
  const auto& t = working_tensors();
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(t[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Qwen3DecoderLoader::merge_host_at_weights() {
  auto& t = working_tensors();
  const bool enable_mc2 =
      ::xllm::KernelConfig::get_instance().enable_fused_mc2() > 0 &&
      ::xllm::ParallelConfig::get_instance().communication_backend() ==
          "hccl" &&
      quantize_type_.empty();

  if (quantize_type_.compare("w8a8") == 0) {
    // Detect if o_proj is non-quantized (bf16) by checking quant_bias numel
    // If IN_ATTENTION_OUT_BIAS is placeholder (numel <= 1), o_proj is bf16
    o_proj_quantized_ = t[IN_ATTENTION_OUT_BIAS].numel() > 1;

    if (o_proj_quantized_) {
      t[IN_ATTENTION_OUT_DEQSCALE] =
          t[IN_ATTENTION_OUT_DEQSCALE].to(torch::kFloat32);
      t[IN_ATTENTION_OUT_OFFSET] = t[IN_ATTENTION_OUT_OFFSET].to(torch::kInt8);
    }

    t[IN_Q_DEQSCALE] =
        torch::cat({t[IN_Q_DEQSCALE], t[IN_K_DEQSCALE], t[IN_V_DEQSCALE]}, 0)
            .to(torch::kFloat32);

    t[IN_Q_BIAS] = torch::cat({t[IN_Q_BIAS], t[IN_K_BIAS], t[IN_V_BIAS]}, 0)
                       .to(torch::kInt32);

    for (auto idx : {IN_K_DEQSCALE,
                     IN_V_DEQSCALE,
                     IN_K_BIAS,
                     IN_V_BIAS,
                     IN_K_OFFSET,
                     IN_V_OFFSET,
                     IN_K_SCALE,
                     IN_V_SCALE}) {
      t[idx] = at_placeholder_;
    }

    t[IN_MLP_W2_BIAS] = torch::cat({t[IN_MLP_W2_BIAS], t[IN_MLP_W1_BIAS]}, 0);

    t[IN_MLP_W2_DEQSCALE] =
        torch::cat({t[IN_MLP_W2_DEQSCALE], t[IN_MLP_W1_DEQSCALE]}, 0)
            .to(torch::kFloat32);

    for (auto idx : {IN_MLP_W1_BIAS,
                     IN_MLP_W1_OFFSET,
                     IN_MLP_W1_SCALE,
                     IN_MLP_W1_DEQSCALE}) {
      t[idx] = at_placeholder_;
    }

    t[IN_Q_OFFSET] = t[IN_Q_OFFSET].to(torch::kInt8);
    t[IN_MLP_W2_OFFSET] = t[IN_MLP_W2_OFFSET].to(torch::kInt8);

    if (t[IN_MLP_CPROJ_BIAS].numel() > 1) {
      t[IN_MLP_CPROJ_BIAS] = t[IN_MLP_CPROJ_BIAS].to(torch::kInt32);
      t[IN_MLP_CPROJ_DEQSCALE] = t[IN_MLP_CPROJ_DEQSCALE].to(torch::kFloat32);
      t[IN_MLP_CPROJ_OFFSET] = t[IN_MLP_CPROJ_OFFSET].to(torch::kInt8);
      t[IN_MLP_CPROJ_SCALE] = t[IN_MLP_CPROJ_SCALE].to(dtype_);
      down_proj_quantized_ = true;
    }

    if (rank_id_ != 0) {
      t[IN_ATTENTION_OUT_BIAS] = torch::zeros_like(t[IN_ATTENTION_OUT_BIAS]);
      if (down_proj_quantized_) {
        t[IN_MLP_CPROJ_BIAS] = torch::zeros_like(t[IN_MLP_CPROJ_BIAS]);
      }
    }
  }

  t[IN_Q_WEIGHT] =
      cast_nz(torch::cat({t[IN_Q_WEIGHT], t[IN_K_WEIGHT], t[IN_V_WEIGHT]}, 0)
                  .transpose(0, 1),
              IN_Q_WEIGHT);

  t[IN_ATTENTION_OUT_WEIGHT] =
      enable_mc2 ? t[IN_ATTENTION_OUT_WEIGHT].transpose(0, 1).contiguous()
                 : cast_nz(t[IN_ATTENTION_OUT_WEIGHT].transpose(0, 1),
                           IN_ATTENTION_OUT_WEIGHT);

  t[IN_MLP_W2_WEIGHT] = cast_nz(
      torch::cat({t[IN_MLP_W2_WEIGHT], t[IN_MLP_W1_WEIGHT]}, 0).transpose(0, 1),
      IN_MLP_W2_WEIGHT);

  t[IN_MLP_CPROJ_WEIGHT] =
      enable_mc2 ? t[IN_MLP_CPROJ_WEIGHT].transpose(0, 1).contiguous()
                 : cast_nz(t[IN_MLP_CPROJ_WEIGHT].transpose(0, 1),
                           IN_MLP_CPROJ_WEIGHT);

  for (auto idx :
       {IN_MLP_W1_WEIGHT, IN_K_WEIGHT, IN_V_WEIGHT, IN_K_BIAS, IN_V_BIAS}) {
    t[idx] = at_placeholder_;
  }

  if (enableAddNorm_) {
    if (quantize_type_.compare("w8a8") == 0) {
      torch::ScalarType weight_fill_dtype = torch::kBFloat16;
      int64_t hidden_size = t[IN_NORM_WEIGHT].size(0);
      t[IN_QKV_SCALE_FILL] =
          t[IN_Q_SCALE].repeat(hidden_size).to(weight_fill_dtype);
      t[IN_MLP_SCALE_FILL] =
          t[IN_MLP_W2_SCALE].repeat(hidden_size).to(weight_fill_dtype);
      t[IN_QKV_OFFSET_FILL] =
          t[IN_Q_OFFSET].repeat(hidden_size).to(weight_fill_dtype);
      t[IN_MLP_OFFSET_FILL] =
          t[IN_MLP_W2_OFFSET].repeat(hidden_size).to(weight_fill_dtype);
    } else {
      for (auto idx : {IN_QKV_SCALE_FILL,
                       IN_QKV_OFFSET_FILL,
                       IN_MLP_SCALE_FILL,
                       IN_MLP_OFFSET_FILL}) {
        t[idx] = at_placeholder_;
      }
    }
  }
}

}  // namespace layer
}  // namespace xllm
