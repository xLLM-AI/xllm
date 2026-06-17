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

#pragma once

#include <torch/torch.h>

#include <optional>
#include <string>

#include "core/framework/state_dict/utils.h"
#include "core/layers/common/add_matmul.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"

namespace xllm::dit {

// Supported linear parallelism strategies.
enum class LinearType {
  // No parallelism — single-device matmul via AddMatmulWeightTransposed.
  Default,
  // Ulysses-style sequence parallelism with all2all communication.
  SequenceParallel,
  // Megatron-style tensor parallelism with column/row splitting +
  // gather/reduce.
  TensorParallel,
};

// ── Sequence Parallel Options ──────────────────────────────────────────────
//
// Configures all2all communication for Ulysses-style sequence parallelism.
//
// If before_attention=true:  linear → all2all (scatter heads, gather seq)
// If before_attention=false: all2all → linear (gather heads, scatter seq)
struct SpOptions {
  int64_t head_num = 0;
  int64_t head_dim = 0;
  int64_t hidden_size = 0;
  bool before_attention = false;
  ProcessGroup* process_group = nullptr;

  SpOptions() = default;

  SpOptions(int64_t head_num,
            int64_t head_dim,
            int64_t hidden_size,
            bool before_attention,
            ProcessGroup* process_group = nullptr)
      : head_num(head_num),
        head_dim(head_dim),
        hidden_size(hidden_size),
        before_attention(before_attention),
        process_group(process_group) {}

  void validate() const {
    CHECK(head_num > 0) << "SpOptions: head_num must be > 0, got " << head_num;
    CHECK(head_dim > 0) << "SpOptions: head_dim must be > 0, got " << head_dim;
    CHECK(hidden_size > 0) << "SpOptions: hidden_size must be > 0, got "
                           << hidden_size;
    CHECK(hidden_size == head_dim * head_num)
        << "SpOptions: hidden_size (" << hidden_size
        << ") must equal head_dim (" << head_dim << ") * head_num (" << head_num
        << ")";
    if (!process_group) {
      LOG(ERROR) << "SpOptions: process_group is nullptr — "
                 << "all2all communication requires a valid process group";
    }
  }
};

// ── Tensor Parallel Options ────────────────────────────────────────────────
//
// Configures Megatron-style tensor parallelism.
//
// Column parallel: weight is split along dim 0 (output features).
//   Each rank holds out_features/tp_size rows. Optionally gathers output.
//
// Row parallel: weight is split along dim 1 (input features).
//   Each rank holds in_features/tp_size columns. Reduces output across ranks.
struct TpOptions {
  bool column_parallel = true;
  int64_t tp_rank = 0;
  int64_t tp_size = 1;
  bool gather_output = false;
  bool need_scatter = false;
  ProcessGroup* process_group = nullptr;

  TpOptions() = default;

  TpOptions(bool column_parallel,
            int64_t tp_rank,
            int64_t tp_size,
            bool gather_output = false,
            bool need_scatter = false,
            ProcessGroup* process_group = nullptr)
      : column_parallel(column_parallel),
        tp_rank(tp_rank),
        tp_size(tp_size),
        gather_output(gather_output),
        need_scatter(need_scatter),
        process_group(process_group) {}

  void validate() const {
    CHECK(tp_size > 0) << "TpOptions: tp_size must be > 0, got " << tp_size;
    CHECK(tp_rank >= 0 && tp_rank < tp_size)
        << "TpOptions: tp_rank (" << tp_rank
        << ") must be in [0, tp_size=" << tp_size << ")";
    if (!process_group) {
      LOG(ERROR) << "TpOptions: process_group is nullptr — "
                 << "tensor parallel communication requires a valid process "
                    "group";
    }
  }
};

class DiTParallelLinearImpl : public torch::nn::Module {
 public:
  DiTParallelLinearImpl(layer::AddMatmulWeightTransposed linear,
                        const std::string& module_name,
                        LinearType linear_type = LinearType::Default,
                        const SpOptions& sp_options = SpOptions())
      : in_features_(0),
        out_features_(0),
        has_bias_(false),
        linear_type_(linear_type),
        sp_options_(sp_options),
        tp_options_(std::nullopt) {
    linear_ = register_module(module_name, std::move(linear));
    if (linear_type_ == LinearType::SequenceParallel) {
      sp_options_.validate();
    }
  }

  DiTParallelLinearImpl(
      int64_t in_features,
      int64_t out_features,
      bool bias,
      const torch::TensorOptions& options,
      LinearType linear_type = LinearType::Default,
      const std::optional<SpOptions>& sp_options = std::nullopt,
      const std::optional<TpOptions>& tp_options = std::nullopt)
      : in_features_(in_features),
        out_features_(out_features),
        has_bias_(bias),
        tensor_options_(options),
        linear_type_(linear_type),
        sp_options_(sp_options.value_or(SpOptions())),
        tp_options_(tp_options) {
    switch (linear_type_) {
      case LinearType::Default:
      case LinearType::SequenceParallel:
        linear_ =
            register_module("linear",
                            layer::AddMatmulWeightTransposed(
                                in_features, out_features, bias, options));
        if (linear_type_ == LinearType::SequenceParallel) {
          sp_options_.validate();
        }
        break;
      case LinearType::TensorParallel:
        CHECK(tp_options_.has_value())
            << "DiTParallelLinear: TpOptions required for TensorParallel mode";
        tp_options_->validate();
        init_tp_weights();
        break;
    }
  }

  torch::Tensor forward(const torch::Tensor& input) {
    switch (linear_type_) {
      case LinearType::Default:
        return linear_->forward(input);
      case LinearType::SequenceParallel:
        return forward_sp(input);
      case LinearType::TensorParallel:
        return forward_tp(input);
      default:
        LOG(FATAL) << "DiTParallelLinear: unknown LinearType "
                   << static_cast<int64_t>(linear_type_);
        return input;
    }
  }

  void load_state_dict(const StateDict& state_dict) {
    if (linear_type_ == LinearType::TensorParallel) {
      load_tp_weights(state_dict);
    } else {
      linear_->load_state_dict(state_dict);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    if (linear_type_ == LinearType::TensorParallel) {
      CHECK(tp_weight_loaded_)
          << "DiTParallelLinear: weight not loaded for " << prefix << "weight";
      if (has_bias_) {
        CHECK(tp_bias_loaded_)
            << "DiTParallelLinear: bias not loaded for " << prefix << "bias";
      }
    } else {
      linear_->verify_loaded_weights(prefix);
    }
  }

  torch::Tensor get_weight() const {
    return linear_type_ == LinearType::TensorParallel ? tp_weight_
                                                      : torch::Tensor();
  }

 private:
  torch::Tensor forward_sp(const torch::Tensor& input) {
    CHECK(input.dim() == 3)
        << "SP linear expects 3D input {batch, seq, hidden}, got shape "
        << input.sizes();

    const auto group_size = sp_options_.process_group->world_size();

    if (sp_options_.before_attention) {
      auto out = linear_->forward(input);
      auto fn = parallel_state::all_to_all_4D(
          out.view(
              {input.size(0), -1, sp_options_.head_num, sp_options_.head_dim}),
          /*scatter_dim=*/2,
          /*gather_dim=*/1,
          /*async=*/false,
          sp_options_.process_group);
      return fn().view(
          {input.size(0), -1, sp_options_.hidden_size / group_size});
    } else {
      auto fn = parallel_state::all_to_all_4D(
          input.view({input.size(0),
                      -1,
                      sp_options_.head_num / group_size,
                      sp_options_.head_dim}),
          /*scatter_dim=*/1,
          /*gather_dim=*/2,
          /*async=*/false,
          sp_options_.process_group);
      auto gathered = fn().view({input.size(0), -1, sp_options_.hidden_size});
      return linear_->forward(gathered);
    }
  }

  void init_tp_weights() {
    const auto& tp = tp_options_.value();
    if (tp.column_parallel) {
      int64_t out_per_partition = out_features_ / tp.tp_size;
      tp_weight_ = register_parameter(
          "weight",
          torch::empty({out_per_partition, in_features_}, tensor_options_),
          /*is_buffer=*/false);
      if (has_bias_) {
        tp_bias_ = register_parameter(
            "bias",
            torch::empty({out_per_partition}, tensor_options_),
            /*is_buffer=*/false);
      }
    } else {
      int64_t in_per_partition = in_features_ / tp.tp_size;
      tp_weight_ = register_parameter(
          "weight",
          torch::empty({out_features_, in_per_partition}, tensor_options_),
          /*is_buffer=*/false);
      if (has_bias_) {
        tp_bias_ =
            register_parameter("bias",
                               torch::empty({out_features_}, tensor_options_),
                               /*is_buffer=*/false);
      }
    }
  }

  torch::Tensor forward_tp(const torch::Tensor& input) {
    const auto& tp = tp_options_.value();
    if (tp.tp_size <= 1) {
      return linear_->forward(input);
    }
    return tp.column_parallel ? forward_tp_column(input)
                              : forward_tp_row(input);
  }

  torch::Tensor forward_tp_column(const torch::Tensor& input) {
    const auto& tp = tp_options_.value();

    auto bias =
        has_bias_ ? std::optional<torch::Tensor>(tp_bias_) : std::nullopt;
    xllm::kernel::MatmulParams params;
    params.a = input;
    params.b = tp_weight_;
    params.bias = bias;
    auto output = xllm::kernel::matmul(params);

    if (tp.gather_output) {
      output = parallel_state::gather(output, tp.process_group, /*dim=*/-1);
    }
    return output;
  }

  torch::Tensor forward_tp_row(const torch::Tensor& input) {
    const auto& tp = tp_options_.value();

    auto x = input;
    if (tp.need_scatter) {
      x = parallel_state::scatter(input, tp.process_group, /*dim=*/-1);
    }

    xllm::kernel::MatmulParams params;
    params.a = x;
    params.b = tp_weight_;
    auto output = xllm::kernel::matmul(params);

    auto orig_dtype = output.dtype();
    auto output_fp32 = output.to(torch::kFloat32);
    output =
        parallel_state::reduce(output_fp32, tp.process_group).to(orig_dtype);

    if (has_bias_) {
      output = output + tp_bias_;
    }
    return output;
  }

  void load_tp_weights(const StateDict& state_dict) {
    const auto& tp = tp_options_.value();
    if (tp.column_parallel) {
      weight::load_sharded_weight(state_dict,
                                  "weight",
                                  /*axis=*/0,
                                  tp.tp_rank,
                                  tp.tp_size,
                                  tp_weight_,
                                  tp_weight_loaded_);
      if (has_bias_) {
        weight::load_sharded_weight(state_dict,
                                    "bias",
                                    /*axis=*/0,
                                    tp.tp_rank,
                                    tp.tp_size,
                                    tp_bias_,
                                    tp_bias_loaded_);
      }
    } else {
      weight::load_sharded_weight(state_dict,
                                  "weight",
                                  /*axis=*/1,
                                  tp.tp_rank,
                                  tp.tp_size,
                                  tp_weight_,
                                  tp_weight_loaded_);
      if (has_bias_) {
        weight::load_weight(state_dict, "bias", tp_bias_, tp_bias_loaded_);
      }
    }
  }

  int64_t in_features_;
  int64_t out_features_;
  bool has_bias_;
  torch::TensorOptions tensor_options_;
  layer::AddMatmulWeightTransposed linear_{nullptr};
  LinearType linear_type_;
  SpOptions sp_options_;
  std::optional<TpOptions> tp_options_;
  torch::Tensor tp_weight_;
  torch::Tensor tp_bias_;
  bool tp_weight_loaded_ = false;
  bool tp_bias_loaded_ = false;
};

TORCH_MODULE(DiTParallelLinear);

}  // namespace xllm::dit
