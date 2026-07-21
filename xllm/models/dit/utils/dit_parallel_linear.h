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

#include <torch/torch.h>

#include <optional>
#include <string>

#include "core/framework/state_dict/utils.h"
#include "core/layers/common/add_matmul.h"
#include "core/layers/common/quant_utils.h"
#include "core/layers/common/rms_norm.h"
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
  // Combined TP+SP: TP shards weights (column/row) while SP handles
  // all2all communication for sequence/head distribution.
  TensorAndSequenceParallel,
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
  bool gather_output = false;
  bool need_scatter = false;
  ProcessGroup* process_group = nullptr;

  TpOptions() = default;

  TpOptions(bool column_parallel,
            bool gather_output,
            bool need_scatter,
            ProcessGroup* process_group)
      : column_parallel(column_parallel),
        gather_output(gather_output),
        need_scatter(need_scatter),
        process_group(process_group) {}

  int64_t tp_size() const {
    return process_group ? process_group->world_size() : 1;
  }
  int64_t tp_rank() const { return process_group ? process_group->rank() : 0; }

  static TpOptions col(ProcessGroup* pg, bool gather_output = false) {
    return TpOptions(
        /*column_parallel=*/true, gather_output, /*need_scatter=*/false, pg);
  }
  static TpOptions row(ProcessGroup* pg, bool gather_output = true) {
    return TpOptions(
        /*column_parallel=*/false, gather_output, /*need_scatter=*/false, pg);
  }

  void validate() const {
    CHECK(tp_size() > 0) << "TpOptions: tp_size must be > 0, got " << tp_size();
    CHECK(tp_rank() >= 0 && tp_rank() < tp_size())
        << "TpOptions: tp_rank (" << tp_rank()
        << ") must be in [0, tp_size=" << tp_size() << ")";
    if (!process_group) {
      LOG(ERROR) << "TpOptions: process_group is nullptr — "
                 << "tensor parallel communication requires a valid process "
                    "group";
    }
  }
};

class DiTParallelLinearImpl : public torch::nn::Module {
 public:
  DiTParallelLinearImpl(
      int64_t in_features,
      int64_t out_features,
      bool bias,
      const torch::TensorOptions& options,
      const std::optional<SpOptions>& sp_options = std::nullopt,
      const std::optional<TpOptions>& tp_options = std::nullopt,
      const QuantArgs& quant_args = QuantArgs())
      : in_features_(in_features),
        out_features_(out_features),
        has_bias_(bias),
        tensor_options_(options),
        linear_type_(infer_linear_type(sp_options, tp_options)),
        sp_options_(sp_options.value_or(SpOptions())),
        tp_options_(tp_options),
        quant_args_(quant_args),
        output_dtype_(c10::typeMetaToScalarType(options.dtype())) {
    if (linear_type_ == LinearType::SequenceParallel ||
        linear_type_ == LinearType::TensorAndSequenceParallel) {
      sp_options_.validate();
    }
    if (has_tp_weights()) {
      CHECK(tp_options_.has_value())
          << "DiTParallelLinear: TpOptions required for the current LinearType";
      tp_options_->validate();
      init_tp_weights();
    } else {
      linear_ = register_module(
          "linear",
          layer::AddMatmulWeightTransposed(
              in_features, out_features, bias, options, quant_args_));
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
      case LinearType::TensorAndSequenceParallel:
        return forward_tp_sp(input);
      default:
        LOG(FATAL) << "DiTParallelLinear: unknown LinearType "
                   << static_cast<int32_t>(linear_type_);
        return input;
    }
  }

  void load_state_dict(const StateDict& state_dict) {
    if (has_tp_weights()) {
      load_tp_weights(state_dict);
    } else {
      linear_->load_state_dict(state_dict);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    if (has_tp_weights()) {
      CHECK(tp_weight_loaded_)
          << "DiTParallelLinear: weight not loaded for " << prefix << "weight";
      if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
        CHECK(tp_weight_scale_is_loaded_)
            << "DiTParallelLinear: weight_scale not loaded for " << prefix
            << "weight_scale";
      }
      if (has_bias_) {
        CHECK(tp_bias_loaded_)
            << "DiTParallelLinear: bias not loaded for " << prefix << "bias";
      }
    } else {
      linear_->verify_loaded_weights(prefix);
    }
  }

  torch::Tensor get_weight() const {
    return has_tp_weights() ? tp_weight_ : torch::Tensor();
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
      int64_t out_per_partition = out_features_ / tp.tp_size();
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
      int64_t in_per_partition = in_features_ / tp.tp_size();
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
    // Pre-allocate 1D weight_scale and weight_offset for dynamic W8A8.
    // Size is always out_features per partition (size(0) of tp_weight_).
    if (!quant_args_.quant_descs().empty()) {
      int64_t scale_size = tp_weight_.size(0);
      tp_weight_scale_ = register_parameter(
          "weight_scale",
          torch::empty({scale_size}, tensor_options_.dtype(torch::kFloat32)),
          /*is_buffer=*/false);
      tp_weight_offset_ = register_parameter(
          "weight_offset",
          torch::empty({scale_size}, tensor_options_.dtype(torch::kFloat32)),
          /*is_buffer=*/false);
    }
  }

  torch::Tensor forward_tp(const torch::Tensor& input) {
    const auto& tp = tp_options_.value();
    if (tp.tp_size() <= 1) {
      return linear_->forward(input);
    }
    return tp.column_parallel ? forward_tp_column(input)
                              : forward_tp_row(input);
  }

  torch::Tensor forward_tp_column(const torch::Tensor& input) {
    const auto& tp = tp_options_.value();
    auto bias =
        has_bias_ ? std::optional<torch::Tensor>(tp_bias_) : std::nullopt;

    torch::Tensor output;
    if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
      CHECK(tp_weight_scale_is_loaded_ && tp_weight_scale_.defined())
          << "tp_weight_scale is required for w8a8_dynamic TP column";
      auto w_offset =
          (tp_weight_offset_.defined() && output_dtype_ == torch::kInt8)
              ? std::optional<torch::Tensor>(tp_weight_offset_)
              : std::nullopt;
      output = layer::npu_w8a8_dynamic_linear_forward(
          input, tp_weight_, tp_weight_scale_, bias, output_dtype_, w_offset);
    } else {
      xllm::kernel::MatmulParams params;
      params.a = input;
      params.b = tp_weight_;
      params.bias = bias;
      output = xllm::kernel::matmul(params);
    }

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

    // Bias must be added AFTER all-reduce for row-parallel; otherwise each
    // rank adds the full bias and reduce-sum multiplies it by tp_size.
    torch::Tensor output;
    if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
      CHECK(tp_weight_scale_is_loaded_ && tp_weight_scale_.defined())
          << "tp_weight_scale is required for w8a8_dynamic TP row";
      auto w_offset =
          (tp_weight_offset_.defined() && output_dtype_ == torch::kInt8)
              ? std::optional<torch::Tensor>(tp_weight_offset_)
              : std::nullopt;
      output = layer::npu_w8a8_dynamic_linear_forward(x,
                                                      tp_weight_,
                                                      tp_weight_scale_,
                                                      /*bias=*/std::nullopt,
                                                      output_dtype_,
                                                      w_offset);
    } else {
      xllm::kernel::MatmulParams params;
      params.a = x;
      params.b = tp_weight_;
      output = xllm::kernel::matmul(params);
    }

    output = parallel_state::reduce(output, tp.process_group);

    if (has_bias_) {
      output = output + tp_bias_;
    }
    return output;
  }

  torch::Tensor forward_tp_sp(const torch::Tensor& input) {
    CHECK(input.dim() == 3)
        << "TP+SP linear expects 3D input {batch, seq, hidden}, got shape "
        << input.sizes();

    const auto& tp = tp_options_.value();
    const auto group_size = sp_options_.process_group->world_size();

    if (tp.column_parallel && sp_options_.before_attention) {
      // TP column output: {batch, seq_per_sp, heads / tp_size, dim_head}
      // SP all2all: scatter heads (dim=2), gather seq (dim=1)
      // → {batch, seq_all, heads / (tp_size * sp_size), dim_head}
      auto out = forward_tp_column(input);
      auto fn = parallel_state::all_to_all_4D(
          out.view({input.size(0),
                    -1,
                    sp_options_.head_num / tp.tp_size(),
                    sp_options_.head_dim}),
          /*scatter_dim=*/2,
          /*gather_dim=*/1,
          /*async=*/false,
          sp_options_.process_group);
      return fn().view({input.size(0),
                        -1,
                        sp_options_.hidden_size / (tp.tp_size() * group_size)});
    } else if (!tp.column_parallel && !sp_options_.before_attention) {
      // Input: {batch, seq_per_sp, heads / (tp_size * sp_size), dim_head}
      // SP all2all: scatter seq (dim=1), gather heads (dim=2)
      // → {batch, seq_all, heads / tp_size, dim_head}
      auto fn = parallel_state::all_to_all_4D(
          input.view({input.size(0),
                      -1,
                      sp_options_.head_num / (tp.tp_size() * group_size),
                      sp_options_.head_dim}),
          /*scatter_dim=*/1,
          /*gather_dim=*/2,
          /*async=*/false,
          sp_options_.process_group);
      auto gathered = fn().view(
          {input.size(0), -1, sp_options_.hidden_size / tp.tp_size()});
      return forward_tp_row(gathered);
    } else {
      LOG(FATAL) << "TP+SP: unsupported combination, column_parallel="
                 << tp.column_parallel
                 << ", before_attention=" << sp_options_.before_attention;
      return input;
    }
  }

  static LinearType infer_linear_type(const std::optional<SpOptions>& sp,
                                      const std::optional<TpOptions>& tp) {
    bool has_sp = sp.has_value() && sp->process_group &&
                  sp->process_group->world_size() > 1;
    bool has_tp = tp.has_value() && tp->process_group &&
                  tp->process_group->world_size() > 1;

    if (has_tp && has_sp) {
      return LinearType::TensorAndSequenceParallel;
    }
    if (has_tp) {
      return LinearType::TensorParallel;
    }
    if (has_sp) {
      return LinearType::SequenceParallel;
    }
    return LinearType::Default;
  }

  bool has_tp_weights() const {
    return linear_type_ == LinearType::TensorParallel ||
           linear_type_ == LinearType::TensorAndSequenceParallel;
  }

  void load_tp_weights(const StateDict& state_dict) {
    const auto& tp = tp_options_.value();

    layer::resolve_weight_quant_method_for_linear_load(
        quant_args_, state_dict, nullptr, resolved_weight_quant_method_);

    int64_t weight_axis = tp.column_parallel ? 0 : 1;
    if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
      // Convert tp_weight from original dtype to int8.
      std::vector<weight::LazyParameterSpec> specs;
      specs.push_back(
          weight::LazyParameterSpec{&tp_weight_,
                                    &tp_weight_loaded_,
                                    "weight",
                                    {tp_weight_.size(0), tp_weight_.size(1)},
                                    tensor_options_.dtype(torch::kInt8)});
      weight::ensure_parameter_storage(this, specs);

      int64_t weight_scale_size = tp_weight_.size(0);
      std::vector<weight::LazyParameterSpec> scale_specs;
      scale_specs.push_back(
          weight::LazyParameterSpec{&tp_weight_scale_,
                                    &tp_weight_scale_is_loaded_,
                                    "weight_scale",
                                    {weight_scale_size},
                                    tensor_options_.dtype(torch::kFloat32)});
      scale_specs.push_back(
          weight::LazyParameterSpec{&tp_weight_offset_,
                                    &tp_weight_offset_is_loaded_,
                                    "weight_offset",
                                    {weight_scale_size},
                                    tensor_options_.dtype(torch::kFloat32)});
      weight::ensure_parameter_storage(this, scale_specs);

      weight::load_sharded_weight(state_dict,
                                  "weight",
                                  /*axis=*/weight_axis,
                                  tp.tp_rank(),
                                  tp.tp_size(),
                                  tp_weight_,
                                  tp_weight_loaded_);
      if (tp.column_parallel) {
        weight::load_sharded_weight(state_dict,
                                    "weight_scale",
                                    /*axis=*/0,
                                    tp.tp_rank(),
                                    tp.tp_size(),
                                    tp_weight_scale_,
                                    tp_weight_scale_is_loaded_);
        if (state_dict.has("weight_offset")) {
          weight::load_sharded_weight(state_dict,
                                      "weight_offset",
                                      /*axis=*/0,
                                      tp.tp_rank(),
                                      tp.tp_size(),
                                      tp_weight_offset_,
                                      tp_weight_offset_is_loaded_);
        }
      } else {
        weight::load_weight(state_dict,
                            "weight_scale",
                            tp_weight_scale_,
                            tp_weight_scale_is_loaded_);
        if (state_dict.has("weight_offset")) {
          weight::load_weight(state_dict,
                              "weight_offset",
                              tp_weight_offset_,
                              tp_weight_offset_is_loaded_);
        }
      }
    } else {
      weight::load_sharded_weight(state_dict,
                                  "weight",
                                  /*axis=*/weight_axis,
                                  tp.tp_rank(),
                                  tp.tp_size(),
                                  tp_weight_,
                                  tp_weight_loaded_);
    }

    if (has_bias_) {
      if (tp.column_parallel) {
        weight::load_sharded_weight(state_dict,
                                    "bias",
                                    /*axis=*/0,
                                    tp.tp_rank(),
                                    tp.tp_size(),
                                    tp_bias_,
                                    tp_bias_loaded_);
      } else {
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
  torch::Tensor tp_weight_scale_;
  torch::Tensor tp_weight_offset_;
  bool tp_weight_loaded_ = false;
  bool tp_bias_loaded_ = false;
  bool tp_weight_scale_is_loaded_ = false;
  bool tp_weight_offset_is_loaded_ = false;
  QuantArgs quant_args_;
  std::optional<std::string> resolved_weight_quant_method_;
  at::ScalarType output_dtype_;
};

TORCH_MODULE(DiTParallelLinear);

// ── Shared SP sequence utilities ─────────────────────────────────────────
//
// Split a tensor along a dimension for sequence parallelism: each rank
// gets its own slice via narrow (no copy).
inline torch::Tensor sp_split_sequence(const torch::Tensor& input,
                                       int64_t dim,
                                       ProcessGroup* sp_group) {
  if (!sp_group || sp_group->world_size() <= 1) return input;
  int64_t dim_size = input.size(dim);
  return input.narrow(dim,
                      sp_group->rank() * (dim_size / sp_group->world_size()),
                      dim_size / sp_group->world_size());
}

// Gather a tensor along a dimension for sequence parallelism via
// all-gather + cat on the target dim.
inline torch::Tensor sp_gather_sequence(const torch::Tensor& input,
                                        int64_t dim,
                                        ProcessGroup* sp_group) {
  if (!sp_group || sp_group->world_size() <= 1) return input;
  auto tensor_list = parallel_state::gather(input.contiguous(), sp_group, dim);
  return torch::cat(tensor_list, dim);
}

// TP-aware RMSNorm: performs RMSNorm on a TP-sharded tensor using only
// scalar all-reduce communication, avoiding the all-gather + scatter
// round-trip required by a naive RMSNorm on the full tensor.
//
// x:    [B, L, dim/tp] — TP-sharded Q or K
// norm: RMSNorm module with full weight [dim]
// pg:   TP process group
//
// Communication: allreduce(SUM) on scalar sum-of-squares per token (~6KB)
//   vs. all-gather on full tensor (~16MB). ~2600x reduction.
inline torch::Tensor tp_rms_norm(const torch::Tensor& x,
                                 layer::RMSNorm& norm,
                                 ProcessGroup* pg) {
  int32_t tp_rank = pg->rank();
  int32_t tp_size = pg->world_size();
  int64_t local_dim = x.size(-1);
  float inv_full_dim = 1.0f / static_cast<float>(local_dim * tp_size);

  // Shard the norm weight: [full_dim] → [local_dim]
  auto weight =
      norm->weight().slice(0, tp_rank * local_dim, (tp_rank + 1) * local_dim);

  // Local sum of squares
  auto x_fp32 = x.to(torch::kFloat32);
  auto local_sum_sq = x_fp32.pow(2).sum(/*dim=*/-1, /*keepdim=*/true);

  // All-reduce sum of squares (scalar per token)
  auto global_sum_sq = local_sum_sq.clone();
  pg->allreduce(global_sum_sq);  // SUM

  // Apply: x / sqrt(sum(x²)/N + eps) * w
  auto output = x_fp32 *
                torch::rsqrt(global_sum_sq * inv_full_dim + norm->eps()) *
                weight.to(torch::kFloat32);

  return output.to(x.dtype());
}

}  // namespace xllm::dit
