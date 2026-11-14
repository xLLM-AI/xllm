/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/parallel_state/dit_sp_qkv_domains.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/common/linear.h"
#include "models/dit/transformers/minimax_h3_blocks.h"
#include "models/dit/transformers/minimax_h3_tp_layout.h"
#include "models/dit/utils/minimax_h3_execution_metrics.h"
#include "models/dit/utils/minimax_h3_qkv_layout.h"
#include "models/dit/utils/minimax_h3_qualified_attention.h"

namespace xllm {

// Qualified TP2 x U8 math gathers full logical heads/FFN activations before
// one dense output GEMM. Normalize each complete 128-wide head locally.
// Every collective uses the existing framework process groups and APIs.

// Ulysses all-to-all wrappers. These mirror xllm::sp_all_to_all /
// sp_all_to_all_reverse (transformer_wan.h:117-160) view-for-view and call the
// same framework collective, parallel_state::all_to_all_4D. They are
// redeclared here instead of including transformer_wan.h so that H3 does not
// depend on the wan model header (which pulls in sparse attention, the DiT
// cache mixin and wan's REGISTER_MODEL_ARGS).
//
// `heads` is the GLOBAL physical head count (wan passes model_args.n_heads());
// the wrappers divide by tp_size internally. H3 passes
// kMiniMaxH3PhysicalHeadsPerTp * tp_size = 32 * 2 = 64, so a rank views its
// input as (bs, shard_seq, 32, 128) and branch A of all_to_all_4D leaves it
// with 32 / 8 = 4 local heads over the full sequence.
inline torch::Tensor minimax_h3_sp_all_to_all(
    const torch::Tensor& input,
    int64_t heads,
    int64_t dim_head,
    int64_t tp_size,
    ProcessGroup* sp_group,
    H3QualifiedLaunchSnapshot* launches = nullptr) {
  auto handler = parallel_state::all_to_all_4D(
      input.view({input.size(0), -1, heads / tp_size, dim_head}),
      /*scatter_dim=*/2,
      /*gather_dim=*/1,
      /*async_ops=*/false,
      sp_group);
  if (launches != nullptr) {
    ++launches->qkv_a2a_launches;
  }
  torch::Tensor result = handler();
  if (launches != nullptr) {
    ++launches->qkv_a2a_waits;
  }
  return result.view({input.size(0),
                      -1,
                      heads * dim_head / (tp_size * sp_group->world_size())});
}

inline torch::Tensor minimax_h3_sp_all_to_all_reverse(
    const torch::Tensor& input,
    int64_t heads,
    int64_t dim_head,
    int64_t tp_size,
    ProcessGroup* sp_group,
    H3QualifiedLaunchSnapshot* launches = nullptr) {
  auto handler = parallel_state::all_to_all_4D(
      input.view({input.size(0),
                  -1,
                  heads / (tp_size * sp_group->world_size()),
                  dim_head}),
      /*scatter_dim=*/1,
      /*gather_dim=*/2,
      /*async_ops=*/false,
      sp_group);
  torch::Tensor result = handler();
  if (launches != nullptr) {
    ++launches->inverse_a2a_calls;
  }
  return result.view({input.size(0), -1, heads * dim_head / tp_size});
}

// A size-1 stand-in process group. layer::ColumnParallelLinear dereferences its
// group unconditionally (linear.cpp:529), so a null group cannot be passed to
// it. With world_size() == 1 every framework collective degenerates to identity
// (parallel_state.cpp:96 gather, :188 reduce, :280 all_to_all_4D), so the
// column linears become plain matmuls and the gather+dense out_proj/fc2 gathers
// are no-ops -- which is what the CPU contract tests rely on.
inline ProcessGroup* minimax_h3_resolve_group(
    ProcessGroup* group,
    std::unique_ptr<ProcessGroup>& owned,
    const torch::Device& device) {
  if (group != nullptr) {
    return group;
  }
  owned = std::make_unique<ProcessGroup>(/*rank=*/0, /*world_size=*/1, device);
  return owned.get();
}

inline void minimax_h3_load_replicated_tensor(const torch::Tensor& target,
                                              const torch::Tensor& source,
                                              const std::string& name) {
  if (!source.defined()) {
    throw std::invalid_argument(
        "MiniMax-H3 TP/UAA source is missing replicated tensor `" + name + "`");
  }
  if (target.sizes() != source.sizes() ||
      target.scalar_type() != source.scalar_type()) {
    throw std::invalid_argument(
        "MiniMax-H3 TP/UAA replicated tensor metadata mismatch for `" + name +
        "`");
  }
  torch::NoGradGuard no_grad;
  target.copy_(source.to(target.device()));
}

inline torch::Tensor minimax_h3_module_weight(const torch::nn::Module& module,
                                              const std::string& name) {
  for (const auto& parameter : module.named_parameters()) {
    if (parameter.key() == "weight") {
      return parameter.value();
    }
  }
  throw std::logic_error("MiniMax-H3 module `" + name + "` has no weight");
}

// Reusable Ulysses staging buffers, allocated once by the resident transformer
// owner and shared by all 50 blocks across all 49 forwards. reserve() is
// idempotent for a fixed row count, so a trajectory never reallocates.
class MiniMaxH3UaaWorkspace final {
 public:
  static constexpr int64_t kSlots = 3;

  void reserve(int64_t local_rows, const torch::TensorOptions& options) {
    if (local_rows <= 0) {
      throw std::invalid_argument(
          "MiniMax-H3 UAA workspace requires a positive local row count");
    }
    const torch::TensorOptions bf16 = options.dtype(torch::kBFloat16);
    if (local_rows == local_rows_ && buffers_[0].defined() &&
        buffers_[0].device() == bf16.device()) {
      return;
    }
    const std::vector<int64_t> shape = {
        1, local_rows, kMiniMaxH3PhysicalHeadsPerTp, kMiniMaxH3HeadDim};
    for (torch::Tensor& buffer : buffers_) {
      buffer = torch::zeros(shape, bf16);
    }
    local_rows_ = local_rows;
    ++reallocations_;
  }

  // Buffer-reusing equivalent of minimax_h3_pad_heads_to_physical(): writes the
  // 28 logical heads into `slot` and leaves the 4 dummy heads at zero (they are
  // zeroed once by reserve() and never written again).
  torch::Tensor pad_heads(int64_t slot, const torch::Tensor& logical) {
    if (slot < 0 || slot >= kSlots ||
        !buffers_[static_cast<size_t>(slot)].defined()) {
      throw std::logic_error("MiniMax-H3 UAA workspace slot is not reserved");
    }
    if (!logical.defined() || logical.dim() != 3 ||
        logical.size(0) != local_rows_ ||
        logical.size(1) != kMiniMaxH3LogicalHeadsPerTp ||
        logical.size(2) != kMiniMaxH3HeadDim) {
      throw std::invalid_argument(
          "MiniMax-H3 UAA workspace padding expects [local_rows,28,128]");
    }
    torch::Tensor& buffer = buffers_[static_cast<size_t>(slot)];
    buffer.narrow(2, 0, kMiniMaxH3LogicalHeadsPerTp).copy_(logical);
    return buffer;
  }

  int64_t local_rows() const { return local_rows_; }
  int64_t reallocations() const { return reallocations_; }

 private:
  std::array<torch::Tensor, kSlots> buffers_{};
  int64_t local_rows_ = 0;
  int64_t reallocations_ = 0;
};

// QKV weights retain head-major grouping, giving each TP rank 28 logical
// heads. Activation padding to 32 happens only around the Ulysses transpose.
// After its inverse, trim dummy heads before TP gather and dense projection.
class MiniMaxH3TpUaaAttentionImpl final : public torch::nn::Module {
 public:
  MiniMaxH3TpUaaAttentionImpl(const MiniMaxH3C4Config& config,
                              ProcessGroup* tp_group,
                              ProcessGroup* sp_group,
                              const torch::TensorOptions& options,
                              const DiTSpQkvDomains& qkv_domains = {},
                              bool qkv_overlap = false)
      : num_heads_(config.num_attention_heads),
        head_dim_(config.attention_head_dim),
        hidden_size_(config.hidden_size),
        qkv_domains_(qkv_domains),
        qkv_overlap_(qkv_overlap) {
    const torch::TensorOptions bf16 = options.dtype(torch::kBFloat16);
    tp_group_ =
        minimax_h3_resolve_group(tp_group, owned_tp_group_, options.device());
    sp_group_ =
        minimax_h3_resolve_group(sp_group, owned_sp_group_, options.device());
    tp_size_ = tp_group_->world_size();
    sp_size_ = sp_group_->world_size();
    if (tp_size_ <= 0 || sp_size_ <= 0 || num_heads_ % tp_size_ != 0) {
      throw std::invalid_argument(
          "MiniMax-H3 TP/UAA attention requires 56 heads divisible by tp_size");
    }
    logical_heads_ = num_heads_ / tp_size_;
    if (sp_size_ > 1) {
      // The 28 -> 32 padding adapter is defined only for the locked
      // world16 TP2 x U8 contract.
      if (tp_size_ != kMiniMaxH3TpSize || sp_size_ != kMiniMaxH3SpSize ||
          head_dim_ != kMiniMaxH3HeadDim ||
          logical_heads_ != kMiniMaxH3LogicalHeadsPerTp ||
          !minimax_h3_head_layout_is_valid()) {
        throw std::invalid_argument(
            "MiniMax-H3 U8 head padding requires the locked TP2xU8 "
            "56/28/32/4/128 head layout");
      }
      physical_heads_ = kMiniMaxH3PhysicalHeadsPerTp;
    } else {
      physical_heads_ = logical_heads_;
    }
    if (qkv_overlap_) {
      validate_dit_sp_qkv_domains(qkv_domains_, sp_group_);
    }
    local_heads_ = physical_heads_ / sp_size_;
    // wan passes the GLOBAL head count to sp_all_to_all, which divides by
    // tp_size internally (transformer_wan.h:123, :805).
    global_physical_heads_ = physical_heads_ * tp_size_;

    q_norm_ = register_module(
        "q_norm", MiniMaxH3RMSNorm(head_dim_, config.qk_norm_eps, bf16));
    k_norm_ = register_module(
        "k_norm", MiniMaxH3RMSNorm(head_dim_, config.qk_norm_eps, bf16));
    qkv_proj_ =
        register_module("qkv_proj",
                        layer::ColumnParallelLinear(hidden_size_,
                                                    3 * num_heads_ * head_dim_,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    QuantArgs{},
                                                    tp_group_,
                                                    bf16));
    // Replicated dense out_proj weight [hidden, 56*128]; the TP heads are
    // gathered before this single matmul (see the class comment), so it is
    // never sharded and never all-reduced.
    out_proj_weight_ = register_buffer(
        "out_proj_weight",
        torch::empty({hidden_size_, num_heads_ * head_dim_}, bf16));
  }

  // `cu_seqlens` must cover the rows the attention actually sees: the local
  // row count when sp_size == 1, and local_rows * sp_size (the full packed
  // sequence) after the Ulysses transpose.
  torch::Tensor forward(
      const torch::Tensor& input,
      const std::optional<torch::Tensor>& rope_frequencies,
      const torch::Tensor& cu_seqlens,
      MiniMaxH3UaaWorkspace* workspace = nullptr,
      bool verify_intermediates = false,
      H3QualifiedLaunchSnapshot* qkv_launches = nullptr,
      MiniMaxH3RainfusionExecutionContext* rainfusion = nullptr) {
    verify_loaded_weights();
    if (!input.defined() || input.dim() != 2 ||
        input.scalar_type() != torch::kBFloat16 ||
        input.size(1) != hidden_size_) {
      throw std::invalid_argument(
          "MiniMax-H3 TP/UAA attention input must be BF16 [T,hidden]");
    }
    const int64_t rows = input.size(0);

    torch::Tensor query_full;
    torch::Tensor key_full;
    torch::Tensor value_full;
    if (qkv_overlap_) {
      // Setup and weight layout are fixed before requests. Reject a missing
      // shared workspace before the first asynchronous collective is submitted.
      if (!qkv_weight_is_split_ || workspace == nullptr ||
          workspace->local_rows() != rows) {
        throw std::logic_error(
            "H3 QKV overlap requires split weights and a reserved workspace");
      }
      const torch::Tensor split = qkv_proj_->weight();
      const int64_t inner = logical_heads_ * head_dim_;
      const auto project = [&](int64_t index) {
        return torch::nn::functional::linear(
                   input, split.narrow(0, index * inner, inner))
            .view({rows, logical_heads_, head_dim_});
      };
      const auto launch = [&](int64_t slot,
                              const torch::Tensor& logical,
                              ProcessGroup* domain) {
        auto pending = parallel_state::all_to_all_4D(
            workspace->pad_heads(slot, logical), 2, 1, true, domain);
        if (qkv_launches != nullptr) {
          ++qkv_launches->qkv_a2a_launches;
        }
        return pending;
      };
      std::array<std::function<torch::Tensor()>, 3> pending;
      torch::Tensor query = q_norm_->forward(project(0));
      if (rope_frequencies.has_value()) {
        query = minimax_h3_apply_rope(query, *rope_frequencies);
      }
      pending[0] = launch(0, query, qkv_domains_.q);
      torch::Tensor key = k_norm_->forward(project(1));
      if (rope_frequencies.has_value()) {
        key = minimax_h3_apply_rope(key, *rope_frequencies);
      }
      pending[1] = launch(1, key, qkv_domains_.k);
      pending[2] = launch(2, project(2), qkv_domains_.v);
      std::array<torch::Tensor*, 3> results = {
          &query_full, &key_full, &value_full};
      for (size_t index = 0; index < pending.size(); ++index) {
        *results[index] = pending[index]();
        if (qkv_launches != nullptr) {
          ++qkv_launches->qkv_a2a_waits;
        }
        pending[index] = nullptr;
      }
    } else {
      // Native qualified baseline: one fused grouped projection and the base
      // SP communicator in fixed Q/K/V order. The ordinary native model is
      // independent of this qualified-only module.
      torch::Tensor grouped =
          qkv_proj_->forward(input).view({rows, logical_heads_, 3, head_dim_});
      torch::Tensor query = q_norm_->forward(grouped.select(2, 0).contiguous());
      torch::Tensor key = k_norm_->forward(grouped.select(2, 1).contiguous());
      torch::Tensor value = grouped.select(2, 2).contiguous();
      if (rope_frequencies.has_value()) {
        query = minimax_h3_apply_rope(query, *rope_frequencies);
        key = minimax_h3_apply_rope(key, *rope_frequencies);
      }
      if (sp_size_ == 1) {
        return project_output(
            minimax_h3_segmented_sdpa(query, key, value, cu_seqlens));
      }
      const torch::Tensor physical_query = pad_physical(workspace, 0, query);
      const torch::Tensor physical_key = pad_physical(workspace, 1, key);
      const torch::Tensor physical_value = pad_physical(workspace, 2, value);
      const int64_t physical_inner = physical_heads_ * head_dim_;
      query_full = minimax_h3_sp_all_to_all(
          physical_query.view({1, rows, physical_inner}),
          global_physical_heads_,
          head_dim_,
          tp_size_,
          sp_group_,
          qkv_launches);
      key_full =
          minimax_h3_sp_all_to_all(physical_key.view({1, rows, physical_inner}),
                                   global_physical_heads_,
                                   head_dim_,
                                   tp_size_,
                                   sp_group_,
                                   qkv_launches);
      value_full = minimax_h3_sp_all_to_all(
          physical_value.view({1, rows, physical_inner}),
          global_physical_heads_,
          head_dim_,
          tp_size_,
          sp_group_,
          qkv_launches);
    }

    const int64_t full_rows = rows * sp_size_;
    const std::vector<int64_t> local_head_shape = {
        full_rows, local_heads_, head_dim_};
    torch::Tensor attention =
        minimax_h3_qualified_attention(query_full.view(local_head_shape),
                                       key_full.view(local_head_shape),
                                       value_full.view(local_head_shape),
                                       cu_seqlens,
                                       rainfusion);

    // ── inverse Ulysses transpose, dummy-zero guard, then trim 32 -> 28 ──
    torch::Tensor restored = minimax_h3_sp_all_to_all_reverse(
        attention.view({1, full_rows, local_heads_ * head_dim_}),
        global_physical_heads_,
        head_dim_,
        tp_size_,
        sp_group_,
        qkv_launches);
    restored = restored.view({1, rows, physical_heads_, head_dim_});
    if (verify_intermediates && !minimax_h3_dummy_heads_are_zero(restored)) {
      LOG(FATAL)
          << "MiniMax-H3 dummy heads are nonzero after inverse all-to-all";
    }
    torch::Tensor logical =
        minimax_h3_trim_heads_to_logical(restored.squeeze(0));
    return project_output(logical);
  }

  // The grouped checkpoint weight is loaded verbatim: ColumnParallelLinear
  // chunks dim 0 into tp_size contiguous blocks, and because the layout is
  // head-major grouped, block r is exactly heads [28r, 28r + 28) with their
  // per-head Q/K/V rows -- the Stage-2 logical shard, head-major grouping
  // preserved. No reorder_grouped_qkv is needed on this path.
  void load_state_dict(const StateDict& state_dict) {
    minimax_h3_load_replicated_tensor(
        minimax_h3_module_weight(*q_norm_, "attn.q_norm"),
        state_dict.get_tensor("q_norm.weight"),
        "attn.q_norm.weight");
    minimax_h3_load_replicated_tensor(
        minimax_h3_module_weight(*k_norm_, "attn.k_norm"),
        state_dict.get_tensor("k_norm.weight"),
        "attn.k_norm.weight");
    const torch::Tensor grouped = state_dict.get_tensor("qkv_proj.weight");
    if (!grouped.defined()) {
      throw std::invalid_argument(
          "MiniMax-H3 TP/UAA attention is missing the grouped QKV weight");
    }
    qkv_proj_->load_state_dict(StateDict({{"weight", grouped}}));
    if (qkv_overlap_) {
      torch::NoGradGuard no_grad;
      torch::Tensor weight = qkv_proj_->weight();
      weight.copy_(minimax_h3_split_grouped_qkv_shard(
          weight, logical_heads_, head_dim_));
      qkv_weight_is_split_ = true;
    }
    // out_proj is replicated dense (gather+dense forward), so the full
    // [hidden, 56*128] checkpoint weight is loaded verbatim on every rank --
    // it is NOT dim-1 sharded like a RowParallelLinear would shard it.
    minimax_h3_load_replicated_tensor(
        out_proj_weight_,
        require_tensor(state_dict, "out_proj.weight"),
        "attn.out_proj.weight");
    loaded_ = true;
  }

  void verify_loaded_weights() const {
    if (!loaded_ || !qkv_proj_->is_weight_loaded() ||
        !out_proj_weight_.defined()) {
      throw std::logic_error(
          "MiniMax-H3 TP/UAA attention weights are not completely loaded");
    }
  }

  int64_t tp_size() const { return tp_size_; }
  int64_t sp_size() const { return sp_size_; }
  int64_t logical_heads() const { return logical_heads_; }
  int64_t physical_heads() const { return physical_heads_; }
  int64_t local_heads() const { return local_heads_; }
  ProcessGroup* tp_group() const { return tp_group_; }
  ProcessGroup* sp_group() const { return sp_group_; }
  torch::Tensor qkv_weight() const { return qkv_proj_->weight(); }
  bool qkv_overlap_enabled() const { return qkv_overlap_; }
  bool qkv_weight_is_split() const { return qkv_weight_is_split_; }
  torch::Tensor out_proj_weight() const { return out_proj_weight_; }

 private:
  // Gather this rank's `logical_heads_` attention-output heads to the full 56
  // heads on the TP group, then run one dense [56*128 -> hidden] matmul.
  // This preserves one full-width projection rather than summing BF16 partial
  // products across TP ranks. The world16 numerical gate validates the actual
  // kernel result; equivalent reduction structure alone is not that proof.
  torch::Tensor project_output(const torch::Tensor& local_heads) const {
    torch::Tensor full_heads =
        parallel_state::gather(local_heads, tp_group_, /*dim=*/1);
    return torch::nn::functional::linear(
        full_heads.reshape({full_heads.size(0), num_heads_ * head_dim_}),
        out_proj_weight_);
  }

  static torch::Tensor require_tensor(const StateDict& state_dict,
                                      const std::string& name) {
    const torch::Tensor value = state_dict.get_tensor(name);
    if (!value.defined()) {
      throw std::invalid_argument("MiniMax-H3 TP/UAA attention is missing `" +
                                  name + "`");
    }
    return value;
  }

  // [local_rows, 28, 128] -> [1, local_rows, 32, 128] with 4 zero dummy heads.
  torch::Tensor pad_physical(MiniMaxH3UaaWorkspace* workspace,
                             int64_t slot,
                             const torch::Tensor& logical) const {
    if (workspace != nullptr) {
      return workspace->pad_heads(slot, logical);
    }
    return minimax_h3_pad_heads_to_physical(logical).unsqueeze(0);
  }

  int64_t num_heads_;
  int64_t head_dim_;
  int64_t hidden_size_;
  int64_t tp_size_ = 1;
  int64_t sp_size_ = 1;
  int64_t logical_heads_ = 0;
  int64_t physical_heads_ = 0;
  int64_t local_heads_ = 0;
  int64_t global_physical_heads_ = 0;
  bool loaded_ = false;
  ProcessGroup* tp_group_ = nullptr;
  ProcessGroup* sp_group_ = nullptr;
  std::unique_ptr<ProcessGroup> owned_tp_group_;
  std::unique_ptr<ProcessGroup> owned_sp_group_;
  MiniMaxH3RMSNorm q_norm_{nullptr};
  MiniMaxH3RMSNorm k_norm_{nullptr};
  DiTSpQkvDomains qkv_domains_;
  bool qkv_overlap_ = false;
  bool qkv_weight_is_split_ = false;
  layer::ColumnParallelLinear qkv_proj_{nullptr};
  torch::Tensor out_proj_weight_;
};
TORCH_MODULE(MiniMaxH3TpUaaAttention);

// Each TP rank loads its slice of both FFN halves. Gather the activation
// before the replicated dense down projection to preserve reference arithmetic.
class MiniMaxH3TpUaaMlpImpl final : public torch::nn::Module {
 public:
  MiniMaxH3TpUaaMlpImpl(const MiniMaxH3C4Config& config,
                        ProcessGroup* tp_group,
                        const torch::TensorOptions& options)
      : ffn_hidden_size_(config.ffn_hidden_size) {
    const torch::TensorOptions bf16 = options.dtype(torch::kBFloat16);
    tp_group_ =
        minimax_h3_resolve_group(tp_group, owned_tp_group_, options.device());
    tp_size_ = tp_group_->world_size();
    if (tp_size_ <= 0 || ffn_hidden_size_ % tp_size_ != 0) {
      throw std::invalid_argument(
          "MiniMax-H3 TP/UAA MLP requires ffn divisible by tp_size");
    }
    ffn_per_rank_ = ffn_hidden_size_ / tp_size_;
    fc1_ = register_module("fc1",
                           layer::ColumnParallelLinear(config.hidden_size,
                                                       2 * ffn_hidden_size_,
                                                       /*bias=*/false,
                                                       /*gather_output=*/false,
                                                       QuantArgs{},
                                                       tp_group_,
                                                       bf16));
    // Replicated dense down weight [hidden, ffn]; the TP activation shards are
    // gathered before this single matmul (see the class comment), so it is
    // never dim-1 sharded and never all-reduced.
    fc2_weight_ = register_buffer(
        "fc2_weight",
        torch::empty({config.hidden_size, ffn_hidden_size_}, bf16));
  }

  torch::Tensor forward(const torch::Tensor& input) {
    verify_loaded_weights();
    std::vector<torch::Tensor> up_gate = fc1_->forward(input).chunk(2, -1);
    torch::Tensor activation = up_gate[0] * torch::silu(up_gate[1]);
    torch::Tensor full_activation =
        parallel_state::gather(activation, tp_group_, /*dim=*/-1);
    return torch::nn::functional::linear(full_activation, fc2_weight_);
  }

  // `fc1.weight` arrives in checkpoint order [gate; up] and is reordered to the
  // runtime order [up; gate] exactly like the dense path
  // (minimax_h3_denoiser.h:494). The merged-variable-shard loader then takes
  // this rank's slice of each half and re-concatenates them, which is precisely
  // minimax_h3_slice_gate_up_weight(). fc2 is replicated dense, so the full
  // [hidden, ffn] checkpoint weight is loaded verbatim on every rank.
  void load_state_dict(const StateDict& state_dict) {
    const torch::Tensor fc1 = state_dict.get_tensor("fc1.weight");
    if (!fc1.defined()) {
      throw std::invalid_argument(
          "MiniMax-H3 TP/UAA MLP is missing `fc1.weight`");
    }
    fc1_->load_state_dict(
        StateDict({{"weight", minimax_h3_reorder_gate_up_to_up_gate(fc1)}}),
        /*shard_tensor_count=*/2,
        {ffn_per_rank_, ffn_per_rank_});
    minimax_h3_load_replicated_tensor(fc2_weight_,
                                      require_tensor(state_dict, "fc2.weight"),
                                      "mlp.fc2.weight");
    loaded_ = true;
  }

  void verify_loaded_weights() const {
    if (!loaded_ || !fc1_->is_weight_loaded() || !fc2_weight_.defined()) {
      throw std::logic_error(
          "MiniMax-H3 TP/UAA MLP weights are not completely loaded");
    }
  }

  int64_t ffn_per_rank() const { return ffn_per_rank_; }
  torch::Tensor fc1_weight() const { return fc1_->weight(); }
  torch::Tensor fc2_weight() const { return fc2_weight_; }

 private:
  static torch::Tensor require_tensor(const StateDict& state_dict,
                                      const std::string& name) {
    const torch::Tensor value = state_dict.get_tensor(name);
    if (!value.defined()) {
      throw std::invalid_argument("MiniMax-H3 TP/UAA MLP is missing `" + name +
                                  "`");
    }
    return value;
  }

  int64_t ffn_hidden_size_;
  int64_t tp_size_ = 1;
  int64_t ffn_per_rank_ = 0;
  bool loaded_ = false;
  ProcessGroup* tp_group_ = nullptr;
  std::unique_ptr<ProcessGroup> owned_tp_group_;
  layer::ColumnParallelLinear fc1_{nullptr};
  torch::Tensor fc2_weight_;
};
TORCH_MODULE(MiniMaxH3TpUaaMlp);

// AdaLN conditioning. The projection is the single largest tensor in a block
// (18 * 5376 x 2688), so it is a layer::ColumnParallelLinear on the TP group
// and the shards are reassembled with parallel_state::gather before the
// [M, modality, parameter, hidden] view. The conditioning math itself (silu,
// view, minimax_h3_select_adaln_parameter) is the existing H3 block math.
class MiniMaxH3TpUaaAdaLNImpl final : public torch::nn::Module {
 public:
  MiniMaxH3TpUaaAdaLNImpl(const MiniMaxH3C4Config& config,
                          ProcessGroup* tp_group,
                          int64_t parameter_count,
                          int64_t modality_count,
                          const torch::TensorOptions& options)
      : hidden_size_(config.hidden_size),
        parameter_count_(parameter_count),
        modality_count_(modality_count) {
    const torch::TensorOptions bf16 = options.dtype(torch::kBFloat16);
    tp_group_ =
        minimax_h3_resolve_group(tp_group, owned_tp_group_, options.device());
    tp_size_ = tp_group_->world_size();
    out_features_ = hidden_size_ * parameter_count_ * modality_count_;
    if (tp_size_ <= 0 || out_features_ % tp_size_ != 0) {
      throw std::invalid_argument(
          "MiniMax-H3 TP/UAA AdaLN output is not divisible by tp_size");
    }
    linear_ =
        register_module("linear",
                        layer::ColumnParallelLinear(config.time_embed_dim,
                                                    out_features_,
                                                    /*bias=*/true,
                                                    /*gather_output=*/false,
                                                    QuantArgs{},
                                                    tp_group_,
                                                    bf16));
  }

  torch::Tensor forward(const torch::Tensor& time_embedding) {
    verify_loaded_weights();
    if (!time_embedding.defined() || time_embedding.dim() != 2 ||
        time_embedding.scalar_type() != torch::kFloat32) {
      throw std::invalid_argument(
          "MiniMax-H3 TP/UAA AdaLN time embedding must be FP32 [M,D]");
    }
    torch::Tensor local =
        linear_->forward(torch::silu(time_embedding).to(torch::kBFloat16));
    torch::Tensor full = parallel_state::gather(local, tp_group_, /*dim=*/-1);
    return full.view({time_embedding.size(0),
                      modality_count_,
                      parameter_count_,
                      hidden_size_});
  }

  void load_state_dict(const StateDict& state_dict) {
    linear_->load_state_dict(state_dict.get_dict_with_prefix("linear."));
    loaded_ = true;
  }

  void verify_loaded_weights() const {
    if (!loaded_ || !linear_->is_weight_loaded()) {
      throw std::logic_error(
          "MiniMax-H3 TP/UAA AdaLN weights are not completely loaded");
    }
  }

  torch::Tensor weight() const { return linear_->weight(); }

 private:
  int64_t hidden_size_;
  int64_t parameter_count_;
  int64_t modality_count_;
  int64_t out_features_ = 0;
  int64_t tp_size_ = 1;
  bool loaded_ = false;
  ProcessGroup* tp_group_ = nullptr;
  std::unique_ptr<ProcessGroup> owned_tp_group_;
  layer::ColumnParallelLinear linear_{nullptr};
};
TORCH_MODULE(MiniMaxH3TpUaaAdaLN);

// One resident transformer block: replicated RMSNorms + AdaLN residual
// conditioning (existing H3 math) around the TP2 x U8 attention and MLP.
class MiniMaxH3TpUaaBlockImpl final : public torch::nn::Module {
 public:
  MiniMaxH3TpUaaBlockImpl(const MiniMaxH3C4Config& config,
                          ProcessGroup* tp_group,
                          ProcessGroup* sp_group,
                          const torch::TensorOptions& options,
                          const DiTSpQkvDomains& qkv_domains = {},
                          bool qkv_overlap = false)
      : config_(config) {
    const torch::TensorOptions bf16 = options.dtype(torch::kBFloat16);
    norm1_ = register_module(
        "norm1", MiniMaxH3RMSNorm(config.hidden_size, config.norm_eps, bf16));
    norm2_ = register_module(
        "norm2", MiniMaxH3RMSNorm(config.hidden_size, config.norm_eps, bf16));
    attn_ = register_module(
        "attn",
        MiniMaxH3TpUaaAttention(
            config, tp_group, sp_group, options, qkv_domains, qkv_overlap));
    mlp_ = register_module("mlp", MiniMaxH3TpUaaMlp(config, tp_group, options));
    adaln_proj_ = register_module("adaln_proj",
                                  MiniMaxH3TpUaaAdaLN(config,
                                                      tp_group,
                                                      /*parameter_count=*/6,
                                                      /*modality_count=*/3,
                                                      options));
  }

  MiniMaxH3ResidualBranchTrace forward(
      const torch::Tensor& input,
      const torch::Tensor& time_embedding,
      const torch::Tensor& combined_indices,
      const torch::Tensor& rope_frequencies,
      const torch::Tensor& cu_seqlens,
      MiniMaxH3UaaWorkspace* workspace = nullptr,
      bool verify_intermediates = false,
      H3QualifiedLaunchSnapshot* qkv_launches = nullptr,
      MiniMaxH3RainfusionExecutionContext* rainfusion = nullptr) {
    verify_loaded_weights();
    torch::Tensor parameters = adaln_proj_->forward(time_embedding);
    MiniMaxH3ResidualBranchTrace trace;
    trace.adaln_parameters = parameters;
    trace.shift_msa =
        minimax_h3_select_adaln_parameter(parameters, 0, combined_indices);
    trace.scale_msa =
        minimax_h3_select_adaln_parameter(parameters, 1, combined_indices);
    trace.gate_msa =
        minimax_h3_select_adaln_parameter(parameters, 2, combined_indices);
    trace.shift_mlp =
        minimax_h3_select_adaln_parameter(parameters, 3, combined_indices);
    trace.scale_mlp =
        minimax_h3_select_adaln_parameter(parameters, 4, combined_indices);
    trace.gate_mlp =
        minimax_h3_select_adaln_parameter(parameters, 5, combined_indices);
    trace.norm1_output = norm1_->forward(input);
    trace.attention_input =
        trace.norm1_output * (trace.scale_msa + 1.0) + trace.shift_msa;
    trace.attention_output = attn_->forward(trace.attention_input,
                                            rope_frequencies,
                                            cu_seqlens,
                                            workspace,
                                            verify_intermediates,
                                            qkv_launches,
                                            rainfusion);
    trace.attention_delta = trace.gate_msa * trace.attention_output;
    torch::Tensor after_attention = input + trace.attention_delta;
    trace.norm2_output = norm2_->forward(after_attention);
    trace.mlp_input =
        trace.norm2_output * (trace.scale_mlp + 1.0) + trace.shift_mlp;
    trace.mlp_output = mlp_->forward(trace.mlp_input);
    trace.mlp_delta = trace.gate_mlp * trace.mlp_output;
    trace.output = after_attention + trace.mlp_delta;
    return trace;
  }

  torch::Tensor forward_output_only(
      const torch::Tensor& input,
      const torch::Tensor& time_embedding,
      const torch::Tensor& combined_indices,
      const torch::Tensor& rope_frequencies,
      const torch::Tensor& cu_seqlens,
      MiniMaxH3UaaWorkspace* workspace = nullptr,
      bool verify_intermediates = false,
      H3QualifiedLaunchSnapshot* qkv_launches = nullptr,
      MiniMaxH3RainfusionExecutionContext* rainfusion = nullptr) {
    verify_loaded_weights();
    torch::Tensor parameters = adaln_proj_->forward(time_embedding);
    torch::Tensor after_attention;
    {
      torch::Tensor shift_msa =
          minimax_h3_select_adaln_parameter(parameters, 0, combined_indices);
      torch::Tensor scale_msa =
          minimax_h3_select_adaln_parameter(parameters, 1, combined_indices);
      torch::Tensor gate_msa =
          minimax_h3_select_adaln_parameter(parameters, 2, combined_indices);
      torch::Tensor attention_input =
          norm1_->forward(input) * (scale_msa + 1.0) + shift_msa;
      torch::Tensor attention_output = attn_->forward(attention_input,
                                                      rope_frequencies,
                                                      cu_seqlens,
                                                      workspace,
                                                      verify_intermediates,
                                                      qkv_launches,
                                                      rainfusion);
      after_attention = input + gate_msa * attention_output;
    }
    torch::Tensor shift_mlp =
        minimax_h3_select_adaln_parameter(parameters, 3, combined_indices);
    torch::Tensor scale_mlp =
        minimax_h3_select_adaln_parameter(parameters, 4, combined_indices);
    torch::Tensor gate_mlp =
        minimax_h3_select_adaln_parameter(parameters, 5, combined_indices);
    torch::Tensor mlp_input =
        norm2_->forward(after_attention) * (scale_mlp + 1.0) + shift_mlp;
    return after_attention + gate_mlp * mlp_->forward(mlp_input);
  }

  void load_state_dict(const StateDict& state_dict) {
    minimax_h3_load_replicated_tensor(
        minimax_h3_module_weight(*norm1_, "norm1"),
        require_tensor(state_dict, "norm1.weight"),
        "norm1.weight");
    minimax_h3_load_replicated_tensor(
        minimax_h3_module_weight(*norm2_, "norm2"),
        require_tensor(state_dict, "norm2.weight"),
        "norm2.weight");
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
    adaln_proj_->load_state_dict(
        state_dict.get_dict_with_prefix("adaln_proj."));
    loaded_ = true;
  }

  void verify_loaded_weights() const {
    if (!loaded_) {
      throw std::logic_error(
          "MiniMax-H3 TP/UAA block weights are not completely loaded");
    }
    attn_->verify_loaded_weights();
    mlp_->verify_loaded_weights();
    adaln_proj_->verify_loaded_weights();
  }

  const MiniMaxH3C4Config& config() const { return config_; }
  MiniMaxH3TpUaaAttention attention() const { return attn_; }
  MiniMaxH3TpUaaMlp mlp() const { return mlp_; }
  MiniMaxH3TpUaaAdaLN adaln_projection() const { return adaln_proj_; }

 private:
  static torch::Tensor require_tensor(const StateDict& state_dict,
                                      const std::string& name) {
    const torch::Tensor value = state_dict.get_tensor(name);
    if (!value.defined()) {
      throw std::invalid_argument("MiniMax-H3 TP/UAA block is missing `" +
                                  name + "`");
    }
    return value;
  }

  MiniMaxH3C4Config config_;
  bool loaded_ = false;
  MiniMaxH3RMSNorm norm1_{nullptr};
  MiniMaxH3RMSNorm norm2_{nullptr};
  MiniMaxH3TpUaaAttention attn_{nullptr};
  MiniMaxH3TpUaaMlp mlp_{nullptr};
  MiniMaxH3TpUaaAdaLN adaln_proj_{nullptr};
};
TORCH_MODULE(MiniMaxH3TpUaaBlock);

}  // namespace xllm
