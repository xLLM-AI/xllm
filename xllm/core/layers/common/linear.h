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

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include "common/flash_comm1_context.h"
#include "core/framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {

// extra args for parallel linear behavior.
struct LinearExtraArgs {
  // parameters for fused smoothquant behavior
  std::string act_mode;
  bool is_gated;

  // default constructor
  LinearExtraArgs(const std::string& act_mode_ = "none", bool is_gated_ = false)
      : act_mode(act_mode_), is_gated(is_gated_) {}
};

// Linear layer with column parallelism.
// The linear layer is defined as Y = XA + b. A is parallelized along
// its second dimension as A = [A_1, ..., A_p].
class ColumnParallelLinearImpl : public torch::nn::Module {
 public:
  ColumnParallelLinearImpl(
      int64_t in_features,
      int64_t out_features,
      bool bias,
      bool gather_output,
      const QuantArgs& quant_args,
      ProcessGroup* process_group,
      const torch::TensorOptions& options,
      const LinearExtraArgs& linear_extra_args = LinearExtraArgs(),
      int32_t output_replicas = 1);

  ColumnParallelLinearImpl(const ModelContext& context);

  torch::Tensor forward(torch::Tensor input);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

  // special load_state_dict for fused cases
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string>& prefixes);

  // load_state_dict for merged weights with variable shard sizes
  void load_state_dict(const StateDict& state_dict,
                       int32_t shard_tensor_count,
                       const std::vector<int64_t>& shard_sizes);

  void pretty_print(std::ostream& stream) const {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  torch::Tensor weight() const {
    if (qweight_is_loaded_) {
      return qweight_;
    }
    return weight_;
  }
  torch::Tensor per_channel_scale() const { return per_channel_scale_; }
  torch::Tensor weight_scale() const { return weight_scale_; }
  bool uses_w8a8_dynamic_quant() const;
  torch::Tensor w8a8_dynamic_weight_scale() const;
  std::optional<torch::Tensor> bias() const;
  ProcessGroup* process_group() const { return process_group_; }
  std::optional<torch::Tensor> smooth() const {
    if (smooth_is_loaded_) {
      return smooth_;
    }
    return std::nullopt;
  }

  bool is_weight_loaded() const {
    if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
      return qweight_is_loaded_ && per_channel_scale_is_loaded_ &&
             smooth_is_loaded_;
    }
    return weight_is_loaded_;
  }

  // Get FP8 input scale for fused RMSNorm+FP8 quantization
  std::optional<torch::Tensor> get_input_scale() const;

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features_per_partition, in_features]
  DEFINE_FUSED_WEIGHT(weight);
  DEFINE_FUSED_WEIGHT(qweight);
  DEFINE_FUSED_WEIGHT(per_channel_scale);
  DEFINE_WEIGHT(smooth);
  DEFINE_FUSED_WEIGHT(bias);

  // FP8 quantization parameters
  DEFINE_FUSED_WEIGHT(weight_scale);  // FP8 weight scale
  DEFINE_FUSED_WEIGHT(
      input_scale);  // FP8 input (activation) scale for static quantization

  // NPU static W8A8 parameters.
  DEFINE_FUSED_WEIGHT(input_offset);  // Activation zero-point for npu_quantize.
  DEFINE_FUSED_WEIGHT(deq_scale);     // Weight descale tensor for quant_matmul.
  DEFINE_FUSED_WEIGHT(
      quant_bias);  // Int32 bias consumed directly by quant_matmul.

  // NPU dynamic W8A8 parameters.
  DEFINE_FUSED_WEIGHT(
      weight_offset);  // Kept for checkpoint parity; current dynamic path
                       // follows the Python reference and does not consume it.

  int64_t rank_;
  int64_t world_size_;
  int64_t weight_rank_;
  int64_t weight_world_size_;
  // whether to gather the output
  bool gather_output_;
  at::Device device_;
  // parallel process group
  ProcessGroup* process_group_;

  // quantization args
  QuantArgs quant_args_;
  torch::TensorOptions options_;
  at::ScalarType output_dtype_;
  LinearExtraArgs linear_extra_args_;
  std::optional<std::string> resolved_weight_quant_method_;
};
TORCH_MODULE(ColumnParallelLinear);

class QKVParallelLinearImpl : public torch::nn::Module {
 public:
  QKVParallelLinearImpl(
      int64_t hidden_size,
      int64_t num_heads,
      int64_t num_kv_heads,
      int64_t head_size,
      int64_t num_kv_head_replicas,
      bool bias,
      bool gather_output,
      const ParallelArgs& parallel_args,
      const torch::TensorOptions& options,
      const QuantArgs& quant_args = QuantArgs{},
      const LinearExtraArgs& linear_extra_args = LinearExtraArgs());

  torch::Tensor forward(torch::Tensor input);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string>& prefixes);
  void load_state_dict(const StateDict& state_dict);

  void pretty_print(std::ostream& stream) const {
    stream << name() << " " << weight().sizes() << " " << weight().device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }
  bool is_weight_loaded() const { return weight_is_loaded_; }

  // Accessors for W8A8 dynamic quantization parameters.
  // Used by attention layers to reorder weight_scale/weight_offset
  // when attn_output_gate is enabled.
  torch::Tensor weight_scale() const { return weight_scale_; }
  torch::Tensor weight_offset() const { return weight_offset_; }
  bool is_weight_scale_loaded() const { return weight_scale_is_loaded_; }
  bool is_weight_offset_loaded() const { return weight_offset_is_loaded_; }

  // Get FP8 input scale for fused RMSNorm+FP8 quantization
  // For QKV, returns max of Q/K/V scales (per-tensor)
  std::optional<torch::Tensor> get_input_scale() const;

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features_per_partition, in_features]
  DEFINE_FUSED_WEIGHT(weight);
  DEFINE_FUSED_WEIGHT(bias);

  // FP8 quantization parameters
  DEFINE_FUSED_WEIGHT(weight_scale);  // FP8 weight scale
  DEFINE_FUSED_WEIGHT(
      input_scale);  // FP8 input (activation) scale for static quantization

  // NPU static W8A8 parameters.
  DEFINE_FUSED_WEIGHT(input_offset);  // Activation zero-point for npu_quantize.
  DEFINE_FUSED_WEIGHT(deq_scale);     // Weight descale tensor for quant_matmul.
  DEFINE_FUSED_WEIGHT(
      quant_bias);  // Int32 bias consumed directly by quant_matmul.

  // NPU dynamic W8A8 parameters.
  DEFINE_FUSED_WEIGHT(
      weight_offset);  // Kept for checkpoint parity; current dynamic path
                       // follows the Python reference and does not consume it.

  int64_t rank_;
  int64_t world_size_;
  int64_t hidden_size_;
  int64_t num_heads_;
  int64_t num_kv_heads_;
  int64_t head_size_;
  int64_t num_kv_head_replicas_;
  // whether to gather the output
  bool gather_output_;
  at::Device device_;
  // parallel args
  ParallelArgs parallel_args_;
  torch::TensorOptions options_;
  // quantization args
  QuantArgs quant_args_;
  at::ScalarType output_dtype_;
  std::optional<std::string> resolved_weight_quant_method_;
};
TORCH_MODULE(QKVParallelLinear);

// Linear layer with row parallelism.
//     The linear layer is defined as Y = XA + b. A is parallelized along
//     its first dimension and X along its second dimension as:
//                -   -
//               | A_1 |
//               | .   |
//           A = | .   |       X = [X_1, ..., X_p]
//               | .   |
//               | A_p |
//                -   -
class RowParallelLinearImpl : public torch::nn::Module {
 public:
  RowParallelLinearImpl(
      int64_t in_features,
      int64_t out_features,
      bool bias,
      bool input_is_parallelized,
      bool enable_result_reduction,
      const QuantArgs& quant_args,
      ProcessGroup* process_group,
      const torch::TensorOptions& options,
      const LinearExtraArgs& linear_extra_args = LinearExtraArgs());

  torch::Tensor forward(torch::Tensor input);

  torch::Tensor forward(torch::Tensor input, RowParallelReduceMode reduce_mode);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

  void pretty_print(std::ostream& stream) const {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const {
    if (qweight_is_loaded_) {
      return qweight_;
    }
    return weight_;
  }
  torch::Tensor per_channel_scale() const { return per_channel_scale_; }
  std::optional<torch::Tensor> smooth() const {
    if (smooth_is_loaded_) {
      return smooth_;
    }
    return std::nullopt;
  }
  ProcessGroup* process_group() const { return process_group_; }

 private:
  torch::Tensor forward_impl(torch::Tensor input,
                             RowParallelReduceMode reduce_mode);

  torch::Tensor mmrs_weight_transposed() const;

  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features, in_features_per_partition]
  DEFINE_WEIGHT(weight);
  DEFINE_WEIGHT(qweight);
  DEFINE_WEIGHT(per_channel_scale);
  DEFINE_WEIGHT(smooth);
  DEFINE_WEIGHT(bias);

  // FP8 quantization parameters
  DEFINE_FUSED_WEIGHT(weight_scale);  // FP8 weight scale
  DEFINE_FUSED_WEIGHT(
      input_scale);  // FP8 input (activation) scale for static quantization

  // NPU static W8A8 parameters.
  DEFINE_WEIGHT(input_offset);  // Activation zero-point for npu_quantize.
  DEFINE_WEIGHT(deq_scale);     // Weight descale tensor for quant_matmul.
  DEFINE_WEIGHT(quant_bias);    // Int32 bias consumed directly by quant_matmul.

  // NPU dynamic W8A8 parameters.
  DEFINE_WEIGHT(
      weight_offset);  // Kept for checkpoint parity; current dynamic path
                       // follows the Python reference and does not consume it.

  // whether the input is already parallelized
  bool input_is_parallelized_;

  // whether to reduce the results
  bool enable_result_reduction_;

  // parallel process group
  ProcessGroup* process_group_;

  int64_t rank_;
  int64_t world_size_;

  // quantization args
  QuantArgs quant_args_;
  torch::TensorOptions options_;
  at::ScalarType output_dtype_;
  LinearExtraArgs linear_extra_args_;
  std::optional<std::string> resolved_weight_quant_method_;
  mutable torch::Tensor mmrs_weight_t_;
};
TORCH_MODULE(RowParallelLinear);

class ReplicatedLinearImpl : public torch::nn::Module {
 public:
  ReplicatedLinearImpl(
      int64_t in_features,
      int64_t out_features,
      bool bias,
      const QuantArgs& quant_args,
      const torch::TensorOptions& options,
      const LinearExtraArgs& linear_extra_args = LinearExtraArgs());

  torch::Tensor forward(torch::Tensor input);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

  void pretty_print(std::ostream& stream) const {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }
  bool uses_w8a8_dynamic_quant() const;
  torch::Tensor w8a8_dynamic_weight_scale() const;
  at::ScalarType output_dtype() const;
  std::optional<torch::Tensor> bias() const;

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features, in_features]
  DEFINE_WEIGHT(weight);
  DEFINE_WEIGHT(bias);

  DEFINE_WEIGHT(weight_scale);   // FP8 scale or dynamic W8A8 weight scale.
  DEFINE_WEIGHT(input_scale);    // FP8 input scale or static W8A8 input scale.
  DEFINE_WEIGHT(input_offset);   // Static W8A8 activation zero-point.
  DEFINE_WEIGHT(deq_scale);      // Static W8A8 descale tensor.
  DEFINE_WEIGHT(quant_bias);     // Static W8A8 quant_matmul bias.
  DEFINE_WEIGHT(weight_offset);  // Dynamic W8A8 checkpoint parity placeholder.
  QuantArgs quant_args_;
  torch::TensorOptions options_;
  at::ScalarType output_dtype_;
  std::optional<std::string> resolved_weight_quant_method_;
};
TORCH_MODULE(ReplicatedLinear);

}  // namespace layer
}  // namespace xllm
