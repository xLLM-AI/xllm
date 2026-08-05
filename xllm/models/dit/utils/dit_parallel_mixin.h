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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/model_context.h"
#include "core/framework/parallel_state/process_group.h"
#include "framework/parallel_state/parallel_state.h"

namespace xllm {
namespace dit {

// Mixin providing classifier-free guidance (CFG) parallelism.
//
// Usage:
//   class MyPipeline : public torch::nn::Module,
//                      public dit::CFGParallelMixin { ... };
class CFGParallelMixin {
 public:
  explicit CFGParallelMixin(const DiTModelContext& context)
      : cfg_group_(context.get_parallel_args().dit_cfg_group_) {}

  // forward_fn(is_positive) -> Tensor  —  caller captures embeddings in lambda.
  // Returns {positive_noise_pred, negative_noise_pred}.
  template <typename ForwardFn>
  std::pair<torch::Tensor, torch::Tensor> exec_with_cfg(
      const ForwardFn& forward_fn) const {
    int32_t cfg_size = 1;
    if (cfg_group_ != nullptr) {
      cfg_size = cfg_group_->world_size();
    }

    CHECK(cfg_size == 1 || cfg_size == 2);

    // Serial execution: evaluate positive and negative conditionals one by one.
    if (cfg_size == 1) {
      return {forward_fn(true), forward_fn(false)};
    }

    // CFG parallel (cfg_size == 2): rank 0 → positive, rank 1 → negative,
    // gather + chunk.
    int32_t rank = cfg_group_->rank();
    torch::Tensor noise_pred = forward_fn(rank == 0);
    torch::Tensor gathered =
        parallel_state::gather(noise_pred, cfg_group_, /*dim=*/0);
    auto chunks = torch::chunk(gathered, 2, 0);
    return {chunks[0], chunks[1]};
  }

 private:
  ProcessGroup* cfg_group_ = nullptr;
};

using SequenceParallelTensor = std::pair<torch::Tensor, int64_t>;
using SequenceParallelTensorMap =
    std::unordered_map<std::string, SequenceParallelTensor>;

class SequenceParallelMixin {
 public:
  torch::Tensor pad_tensor(const torch::Tensor& input,
                           const std::string& tensor_name,
                           int64_t dim) const {
    if (!input.defined()) {
      return input;
    }

    const int64_t padding = padding_length(tensor_name);
    if (padding == 0) {
      return input;
    }

    const int64_t normalized_dim = normalize_dim(input, dim);
    std::vector<int64_t> padding_config(static_cast<size_t>(input.dim() * 2),
                                        0);
    const int64_t padding_index = 2 * (input.dim() - normalized_dim - 1) + 1;
    padding_config[static_cast<size_t>(padding_index)] = padding;
    return torch::pad(input, padding_config, "constant", 0);
  }

  torch::Tensor unpad_tensor(const torch::Tensor& input,
                             const std::string& tensor_name,
                             int64_t dim) const {
    if (!input.defined()) {
      return input;
    }

    const int64_t padding = padding_length(tensor_name);
    if (padding == 0) {
      return input;
    }

    const int64_t normalized_dim = normalize_dim(input, dim);
    CHECK_GE(input.size(normalized_dim), padding)
        << "Padding length exceeds tensor size";
    return input.narrow(normalized_dim,
                        /*start=*/0,
                        input.size(normalized_dim) - padding);
  }

 protected:
  explicit SequenceParallelMixin(ProcessGroup* process_group)
      : process_group_(process_group) {}

  template <typename ForwardFn>
  SequenceParallelTensorMap sequence_parallel_forward(
      const SequenceParallelTensorMap& inputs,
      ForwardFn&& forward_fn) {
    padding_lengths_.clear();
    SequenceParallelTensorMap local_inputs = inputs;
    for (auto& [tensor_name, tensor_and_dim] : local_inputs) {
      tensor_and_dim.first = scatter_sequence(
          tensor_and_dim.first, tensor_name, tensor_and_dim.second);
    }

    SequenceParallelTensorMap outputs =
        std::forward<ForwardFn>(forward_fn)(local_inputs);
    for (auto& [tensor_name, tensor_and_dim] : outputs) {
      tensor_and_dim.first = gather_sequence(
          tensor_and_dim.first, tensor_name, tensor_and_dim.second);
    }
    return outputs;
  }

 private:
  int32_t world_size() const {
    return process_group_ == nullptr ? 1 : process_group_->world_size();
  }

  bool sequence_parallel_enabled() const { return world_size() > 1; }

  torch::Tensor scatter_sequence(const torch::Tensor& input,
                                 const std::string& tensor_name,
                                 int64_t sequence_dim) {
    if (!input.defined()) {
      return input;
    }

    const int64_t sequence_length = input.size(sequence_dim);
    const int64_t padding_length =
        (world_size() - sequence_length % world_size()) % world_size();
    padding_lengths_[tensor_name] = padding_length;
    if (!sequence_parallel_enabled()) {
      return input;
    }

    torch::Tensor padded_input = pad_tensor(input, tensor_name, sequence_dim);
    return parallel_state::scatter(
        padded_input, process_group_, static_cast<int32_t>(sequence_dim));
  }

  torch::Tensor gather_sequence(const torch::Tensor& input,
                                const std::string& tensor_name,
                                int64_t sequence_dim) const {
    if (!sequence_parallel_enabled() || !input.defined()) {
      return input;
    }

    torch::Tensor output = parallel_state::gather(
        input.contiguous(), process_group_, static_cast<int32_t>(sequence_dim));
    return unpad_tensor(output, tensor_name, sequence_dim);
  }

  int64_t normalize_dim(const torch::Tensor& input, int64_t dim) const {
    const int64_t normalized_dim = dim < 0 ? input.dim() + dim : dim;
    CHECK_GE(normalized_dim, 0) << "Invalid tensor dimension: " << dim;
    CHECK_LT(normalized_dim, input.dim())
        << "Invalid tensor dimension: " << dim;
    return normalized_dim;
  }

  int64_t padding_length(const std::string& tensor_name) const {
    auto padding_it = padding_lengths_.find(tensor_name);
    CHECK(padding_it != padding_lengths_.end())
        << "Missing sequence-parallel padding metadata: " << tensor_name;
    return padding_it->second;
  }

 protected:
  ProcessGroup* process_group_{nullptr};
  inline static std::unordered_map<std::string, int64_t> padding_lengths_;
};

// Mixin providing 1D spatial-parallel VAE encode/decode.
//
// Splits the global W dimension across vae_size ranks; each rank owns a
// contiguous W-local slice. Conv layers obtain neighbour columns through
// vae_parallel_exchange(). Ops needing the full global tensor use
// vae_parallel_merge() then vae_parallel_split() to restore the local view.
// When vae_size == 1 every op is identity.
//
// Usage:
//   class MyModule : public torch::nn::Module,
//                    public dit::VaeParallelMixin {
//     MyModule(..., const ModelContext& context)
//         : dit::VaeParallelMixin(context) { ... }
//   };
class VaeParallelMixin {
 public:
  explicit VaeParallelMixin(const ModelContext& context)
      : pg_(context.get_parallel_args().dit_vae_group_) {
    CHECK(context.get_parallel_args().vae_size() > 0)
        << "vae_size must be positive";
    CHECK(context.get_parallel_args().vae_size() == 1 || pg_ != nullptr)
        << "ProcessGroup must be provided when vae_size > 1";
    if (pg_ != nullptr) {
      CHECK(context.get_parallel_args().vae_size() == pg_->world_size())
          << "vae_size must equal ProcessGroup world_size";
    }
  }

  // ---- queries ------------------------------------------------------------
  // TODO: move this func to private.
  bool vae_parallel_enabled() const {
    return pg_ != nullptr && pg_->world_size() > 1;
  }

  // ---- operations (identity when !vae_parallel_enabled()) -----------------

  /// Split global tensor along the last (W) dimension into this rank's slice.
  torch::Tensor vae_parallel_split(torch::Tensor x) const {
    if (!vae_parallel_enabled()) {
      return x;
    }
    int64_t width = x.size(-1);
    int64_t base_w = width / vae_size();
    int64_t rem_w = width % vae_size();
    int64_t start_w =
        vae_rank() * base_w + std::min<int64_t>(vae_rank(), rem_w);
    int64_t w_local = base_w + (vae_rank() < rem_w ? 1 : 0);
    return x.slice(/*dim=*/-1, start_w, start_w + w_local).contiguous();
  }

  /// All-gather local slices along W, concatenate back to global tensor.
  torch::Tensor vae_parallel_merge(const torch::Tensor& local_patch) const {
    if (!vae_parallel_enabled()) {
      return local_patch;
    }

    auto orig_sizes = local_patch.sizes();

    // All-gather local widths first (may differ across ranks).
    auto width_options = torch::TensorOptions()
                             .dtype(torch::kInt64)
                             .device(local_patch.device());
    auto local_w = torch::tensor({local_patch.size(-1)}, width_options);
    std::vector<torch::Tensor> w_list(vae_size());
    for (int32_t i = 0; i < vae_size(); ++i) {
      w_list[i] = torch::empty({1}, width_options);
    }
    pg_->allgather(local_w, w_list);

    std::vector<int64_t> widths(vae_size());
    std::vector<std::vector<int64_t>> target_shapes(vae_size());
    for (int32_t i = 0; i < vae_size(); ++i) {
      widths[i] = w_list[i][0].item<int64_t>();
      target_shapes[i] =
          std::vector<int64_t>(orig_sizes.begin(), orig_sizes.end());
      target_shapes[i].back() = widths[i];
    }

    return allgather_variable_width(local_patch, widths, target_shapes);
  }

  /// Halo exchange. If @p pad is true, missing neighbours are zero-padded;
  /// otherwise omitted (for up-sampling trim-exchange).
  torch::Tensor vae_parallel_exchange(torch::Tensor local_patch,
                                      bool pad) const {
    if (!vae_parallel_enabled()) {
      return local_patch;
    }

    auto left_col = local_patch.slice(/*dim=*/-1, 0, 1).contiguous();
    auto right_col =
        local_patch
            .slice(/*dim=*/-1, local_patch.size(-1) - 1, local_patch.size(-1))
            .contiguous();

    const bool has_left = vae_parallel_has_left();
    const bool has_right = vae_parallel_has_right();

    torch::Tensor left_recv, right_recv;
    if (has_left) {
      left_recv = torch::empty_like(right_col);
    }
    if (has_right) {
      right_recv = torch::empty_like(left_col);
    }

    // Even/odd ordering avoids deadlock in blocking send/recv ring.
    const int32_t rank = vae_rank();
    if (rank % 2 == 0) {
      if (has_right) {
        pg_->send(right_col, rank + 1);
        pg_->recv(right_recv, rank + 1);
      }
      if (has_left) {
        pg_->send(left_col, rank - 1);
        pg_->recv(left_recv, rank - 1);
      }
    } else {
      if (has_left) {
        pg_->recv(left_recv, rank - 1);
        pg_->send(left_col, rank - 1);
      }
      if (has_right) {
        pg_->recv(right_recv, rank + 1);
        pg_->send(right_col, rank + 1);
      }
    }

    // torch::cat allocates a new contiguous tensor, no extra contiguous() call.
    if (pad) {
      auto left_pad = has_left ? left_recv : torch::zeros_like(left_col);
      auto right_pad = has_right ? right_recv : torch::zeros_like(right_col);
      return torch::cat({left_pad, local_patch, right_pad}, -1);
    }
    if (!has_left) {
      return torch::cat({local_patch, right_recv}, -1);
    }
    if (!has_right) {
      return torch::cat({left_recv, local_patch}, -1);
    }
    return torch::cat({left_recv, local_patch, right_recv}, -1);
  }

  /// Trim halo columns from the last (W) dimension after an upsample.
  /// Removes @p per_side columns on each side that received a neighbour's
  /// halo: both sides for a middle rank, right-only for the first rank,
  /// left-only for the last rank. Identity when parallelism is disabled.
  torch::Tensor vae_parallel_trim_halo(torch::Tensor x,
                                       int64_t per_side) const {
    if (!vae_parallel_enabled()) {
      return x;
    }
    int64_t cur_w = x.size(-1);
    if (vae_parallel_has_left() && vae_parallel_has_right()) {
      return x.slice(/*dim=*/-1, per_side, cur_w - per_side);
    } else if (!vae_parallel_has_left()) {
      return x.slice(/*dim=*/-1, 0, cur_w - per_side);
    } else {
      return x.slice(/*dim=*/-1, per_side, cur_w);
    }
  }

 private:
  // Valid only when vae_parallel_enabled(); pg_ is null for vae_size == 1.
  int32_t vae_size() const { return pg_->world_size(); }
  int32_t vae_rank() const { return pg_->rank(); }

  bool vae_parallel_has_left() const { return vae_rank() > 0; }
  bool vae_parallel_has_right() const { return vae_rank() < vae_size() - 1; }

  /// All-gather variable-width tensors, then unpad and cat along last dim.
  torch::Tensor allgather_variable_width(
      const torch::Tensor& local,
      const std::vector<int64_t>& widths,
      const std::vector<std::vector<int64_t>>& target_shapes) const {
    int64_t prefix_numel = 1;
    for (int64_t d = 0; d < local.dim() - 1; ++d) {
      prefix_numel *= local.size(d);
    }

    std::vector<int64_t> flat_sizes(vae_size());
    int64_t max_flat_size = 0;
    for (int32_t i = 0; i < vae_size(); ++i) {
      flat_sizes[i] = prefix_numel * widths[i];
      max_flat_size = std::max(max_flat_size, flat_sizes[i]);
    }

    auto local_flat = local.reshape({-1});
    auto padded_flat = torch::zeros({max_flat_size}, local.options());
    padded_flat.slice(/*dim=*/0, 0, local_flat.size(0)).copy_(local_flat);

    std::vector<torch::Tensor> gathered_flat(vae_size());
    for (int32_t i = 0; i < vae_size(); ++i) {
      gathered_flat[i] = torch::empty({max_flat_size}, local.options());
    }
    pg_->allgather(padded_flat, gathered_flat);

    std::vector<torch::Tensor> gathered_tensors;
    gathered_tensors.reserve(vae_size());
    for (int32_t i = 0; i < vae_size(); ++i) {
      gathered_tensors.emplace_back(gathered_flat[i]
                                        .slice(/*dim=*/0, 0, flat_sizes[i])
                                        .reshape(target_shapes[i]));
    }
    return torch::cat(gathered_tensors, -1);
  }

  // Non-owning; lifetime managed by ParallelArgs.
  ProcessGroup* pg_ = nullptr;
};

}  // namespace dit
}  // namespace xllm
