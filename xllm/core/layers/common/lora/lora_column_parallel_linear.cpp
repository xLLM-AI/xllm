/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "layers/common/lora/lora_column_parallel_linear.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "framework/lora/lora_context.h"
#include "framework/lora/lora_runtime.h"

namespace xllm {
namespace layer {

LoRAColumnParallelLinearImpl::LoRAColumnParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const QuantArgs& quant_args,
    ProcessGroup* process_group,
    const torch::TensorOptions& options,
    const std::string& proj_name,
    const LinearExtraArgs& linear_extra_args)
    : proj_name_(proj_name),
      in_features_(in_features),
      out_features_(out_features),
      is_fused_gate_up_(proj_name == "gate_up_proj") {
  base_ = ColumnParallelLinear(in_features,
                               out_features,
                               bias,
                               gather_output,
                               quant_args,
                               process_group,
                               options,
                               linear_extra_args);

  // out_features here is the total; per-rank it is out_features / world_size.
  tp_rank_ = process_group ? process_group->rank() : 0;
  tp_world_size_ = process_group ? process_group->world_size() : 1;
  const int64_t world_size = tp_world_size_;
  out_size_local_ = out_features / std::max<int64_t>(1, world_size);
  inter_size_local_ = is_fused_gate_up_ ? out_size_local_ / 2 : out_size_local_;
}

torch::Tensor LoRAColumnParallelLinearImpl::forward(torch::Tensor input) {
  auto y = base_->forward(input);

  const auto* ctx = current_lora_context();
  if (ctx == nullptr || ctx->adapter_ids == nullptr ||
      ctx->q_seq_lens_vec == nullptr || ctx->layer_index < 0) {
    return y;
  }
  const auto& adapter_ids = *ctx->adapter_ids;
  const auto& q_seq_lens = *ctx->q_seq_lens_vec;
  if (adapter_ids.empty() || adapter_ids.size() != q_seq_lens.size()) {
    return y;
  }
  bool any_nonzero = false;
  for (auto id : adapter_ids) {
    if (id != 0) {
      any_nonzero = true;
      break;
    }
  }
  if (!any_nonzero) return y;

  // P1a Phase 0 fast path: single-adapter batch (99% prod). Skip per-seq
  // slice/matmul loop; do 2 matmuls per proj on full [T, hidden].
  {
    uint64_t sole_aid = 0;
    bool single_adapter = true;
    for (auto id : adapter_ids) {
      if (id == 0) {
        single_adapter = false;  // base-only seq disqualifies fast path
        break;
      }
      if (sole_aid == 0) {
        sole_aid = id;
      } else if (id != sole_aid) {
        single_adapter = false;
        break;
      }
    }
    if (single_adapter && sole_aid != 0) {
      auto& runtime = LoRARuntime::instance();
      if (is_fused_gate_up_) {
        const auto* gate_pd =
            runtime.get_per_proj_delta(sole_aid, ctx->layer_index, "gate_proj");
        const auto* up_pd =
            runtime.get_per_proj_delta(sole_aid, ctx->layer_index, "up_proj");
        if (gate_pd == nullptr && up_pd == nullptr) return y;
        auto shrink_expand =
            [&](const LoRARuntime::ProjDelta* pd) -> torch::Tensor {
          if (pd == nullptr) {
            return torch::zeros({input.size(0), inter_size_local_},
                                input.options());
          }
          auto tmp = torch::matmul(input, pd->A.transpose(0, 1));
          torch::Tensor B_local = pd->B;
          if (tp_world_size_ > 1 && pd->B.size(0) > inter_size_local_) {
            const int64_t start = tp_rank_ * inter_size_local_;
            B_local = pd->B.slice(0, start, start + inter_size_local_);
          }
          auto d = torch::matmul(tmp, B_local.transpose(0, 1));
          return (d * pd->scaling).to(input.dtype());
        };
        auto gate_delta = shrink_expand(gate_pd);
        auto up_delta = shrink_expand(up_pd);
        auto fused_delta = torch::cat({gate_delta, up_delta}, /*dim=*/-1);
        y.add_(fused_delta);
        return y;
      } else {
        // Single-proj branch (reserved). Not exercised in Qwen family
        // currently, but mirror the fast path for symmetry.
        const auto* pd =
            runtime.get_per_proj_delta(sole_aid, ctx->layer_index, proj_name_);
        if (pd == nullptr) return y;
        auto tmp = torch::matmul(input, pd->A.transpose(0, 1));
        torch::Tensor B_local = pd->B;
        if (tp_world_size_ > 1 && pd->B.size(0) > out_size_local_) {
          const int64_t start = tp_rank_ * out_size_local_;
          B_local = pd->B.slice(0, start, start + out_size_local_);
        }
        auto delta = torch::matmul(tmp, B_local.transpose(0, 1));
        delta = (delta * pd->scaling).to(y.dtype());
        y.add_(delta);
        return y;
      }
    }
  }

  auto& runtime = LoRARuntime::instance();
  int64_t tok_off = 0;

  for (size_t seq_idx = 0; seq_idx < adapter_ids.size(); ++seq_idx) {
    const int32_t seq_len = q_seq_lens[seq_idx];
    if (seq_len <= 0) continue;
    const uint64_t aid = adapter_ids[seq_idx];
    if (aid == 0) {
      tok_off += seq_len;
      continue;
    }

    auto x_seq = input.slice(0, tok_off, tok_off + seq_len);

    if (is_fused_gate_up_) {
      // Fused gate_up: lookup gate_proj + up_proj deltas, concat.
      const auto* gate_pd =
          runtime.get_per_proj_delta(aid, ctx->layer_index, "gate_proj");
      const auto* up_pd =
          runtime.get_per_proj_delta(aid, ctx->layer_index, "up_proj");
      if (gate_pd == nullptr && up_pd == nullptr) {
        tok_off += seq_len;
        continue;
      }
      auto make_delta = [&](const LoRARuntime::ProjDelta* pd) -> torch::Tensor {
        if (pd == nullptr) {
          return torch::zeros({x_seq.size(0), inter_size_local_},
                              x_seq.options());
        }
        auto tmp = torch::matmul(x_seq, pd->A.transpose(0, 1));
        // TP shard: adapter's B is [inter_full, rank]. This rank owns
        // rows [tp_rank * inter_size_local_, +inter_size_local_).
        torch::Tensor B_local = pd->B;
        if (tp_world_size_ > 1 && pd->B.size(0) > inter_size_local_) {
          const int64_t start = tp_rank_ * inter_size_local_;
          B_local = pd->B.slice(0, start, start + inter_size_local_);
        }
        auto d = torch::matmul(tmp, B_local.transpose(0, 1));
        return (d * pd->scaling).to(x_seq.dtype());
      };
      auto gate_delta = make_delta(gate_pd);
      auto up_delta = make_delta(up_pd);
      auto fused_delta = torch::cat({gate_delta, up_delta}, /*dim=*/-1);
      y.slice(0, tok_off, tok_off + seq_len).add_(fused_delta);
    } else {
      // Single proj (future use — reserved). Not applicable to Qwen for now.
      const auto* pd =
          runtime.get_per_proj_delta(aid, ctx->layer_index, proj_name_);
      if (pd == nullptr) {
        tok_off += seq_len;
        continue;
      }
      auto tmp = torch::matmul(x_seq, pd->A.transpose(0, 1));
      // TP shard: adapter's B is [out_full, rank]. This rank owns
      // rows [tp_rank * out_size_local_, +out_size_local_).
      torch::Tensor B_local = pd->B;
      if (tp_world_size_ > 1 && pd->B.size(0) > out_size_local_) {
        const int64_t start = tp_rank_ * out_size_local_;
        B_local = pd->B.slice(0, start, start + out_size_local_);
      }
      auto delta = torch::matmul(tmp, B_local.transpose(0, 1));
      delta = (delta * pd->scaling).to(y.dtype());
      y.slice(0, tok_off, tok_off + seq_len).add_(delta);
    }
    tok_off += seq_len;
  }
  return y;
}

void LoRAColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict) {
  base_->load_state_dict(state_dict);
}

void LoRAColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  base_->load_state_dict(state_dict, prefixes);
}

void LoRAColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    int32_t shard_tensor_count,
    const std::vector<int64_t>& shard_sizes) {
  base_->load_state_dict(state_dict, shard_tensor_count, shard_sizes);
}

}  // namespace layer
}  // namespace xllm
