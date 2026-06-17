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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <tuple>

#include "core/framework/state_dict/utils.h"
#include "layers/common/attention_metadata.h"

namespace xllm {
namespace layer {

class CompressorImpl : public torch::nn::Module {
 public:
  CompressorImpl() = default;
  CompressorImpl(int64_t compress_ratio, int64_t head_dim);
  CompressorImpl(
      int64_t compress_ratio,
      int64_t head_dim,
      const torch::TensorOptions& options =
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  CompressorImpl(
      int64_t compress_ratio,
      int64_t head_dim,
      int64_t rope_head_dim,
      int64_t rot_mode,
      double norm_eps,
      const torch::TensorOptions& options =
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

  torch::Tensor forward(const DSAMetadata& attn_metadata,
                        torch::Tensor& hidden_states,
                        std::tuple<torch::Tensor, torch::Tensor>& kv_states,
                        std::tuple<torch::Tensor, torch::Tensor>& block_tables,
                        torch::Tensor& compressed_sin,
                        torch::Tensor& compressed_cos,
                        torch::Tensor actual_seq_lengths_query);

  void load_state_dict(const StateDict& state_dict);
  int64_t weight_bytes() const;

 private:
  torch::Tensor cmp_wkv_;
  torch::Tensor cmp_wgate_;
  torch::Tensor cmp_norm_;
  torch::Tensor cmp_ape_;
  bool cmp_wkv_loaded_ = false;
  bool cmp_wgate_loaded_ = false;
  bool cmp_norm_loaded_ = false;
  bool cmp_ape_loaded_ = false;
  std::string last_load_prefix_;
  int64_t rope_head_dim_;
  int64_t head_dim_;
  int64_t compress_ratio_;
  int64_t rot_mode_;
  double eps_;
  torch::TensorOptions options_;
  bool enable_compressor_overlap_ = false;
};
TORCH_MODULE(Compressor);

}  // namespace layer
}  // namespace xllm
