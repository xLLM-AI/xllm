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

#include "flash_comm1_context.h"

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#if defined(USE_NPU)
#include <torch_npu/torch_npu.h>
#endif

#include "framework/parallel_state/parallel_state.h"

namespace xllm {

namespace {

constexpr int32_t kFc1LocalTokenAlignment = 16;
constexpr size_t kQuantizedMmrsRetainedBytesLimit = 512 * 1024 * 1024;
thread_local const FlashComm1Context* current_flash_comm1_context = nullptr;
thread_local std::vector<torch::Tensor> quantized_mmrs_launch_tensors;
thread_local size_t quantized_mmrs_retained_bytes = 0;

void synchronize_quantized_mmrs_launches() {
  if (quantized_mmrs_launch_tensors.empty()) {
    return;
  }
#if defined(USE_NPU)
  torch::npu::synchronize();
#endif
  quantized_mmrs_launch_tensors.clear();
  quantized_mmrs_retained_bytes = 0;
}

int32_t round_up_to_multiple(int32_t value, int32_t multiple) {
  CHECK_GT(multiple, 0);
  const int32_t remainder = value % multiple;
  return remainder == 0 ? value : value + multiple - remainder;
}

}  // namespace

namespace {

// TP width of the current rank. Prefers the process group, which is
// authoritative, and otherwise derives it from the documented rank layout
// dp_rank * (cp_size * tp_size) + cp_rank * tp_size + tp_rank -- the same
// arithmetic ParallelArgs::cp_rank() uses. The fallback matters because the
// eligibility gate is deliberately callable without a live process group.
int32_t fc1_tp_world_size(const ParallelArgs& parallel_args) {
  if (parallel_args.tp_group_ != nullptr) {
    return parallel_args.tp_group_->world_size();
  }
  const int32_t dp_size = std::max<int32_t>(parallel_args.dp_size(), 1);
  const int32_t cp_size = std::max<int32_t>(parallel_args.cp_size(), 1);
  const int32_t world_size = std::max<int32_t>(parallel_args.world_size(), 1);
  return std::max<int32_t>(world_size / dp_size / cp_size, 1);
}

}  // namespace

bool is_flash_comm1_eligible(const FlashComm1TokenGeometry& geometry,
                             bool is_prefill,
                             const ParallelArgs& parallel_args,
                             const FlashComm1Options& options) {
  if (!options.enable_flashcomm1 || !is_prefill) {
    return false;
  }
  // Threshold on the pre-CP count so the decision is identical on every rank of
  // the CP group.
  if (geometry.global_num_tokens < options.min_prefill_tokens) {
    return false;
  }
  if (std::max<int32_t>(parallel_args.cp_size(), 1) == 1) {
    // No outer CP shard: geometry collapses and the threshold above is the
    // whole gate, exactly as before CP composition existed.
    return true;
  }
  if (geometry.local_num_tokens <= 0) {
    return false;
  }
  // Uneven CP segments: require the smallest one to still carry a full
  // alignment unit per TP rank, otherwise padding would dominate the shard and
  // the thinnest rank could reduce_scatter an all-padding tensor.
  const int32_t min_rows_per_cp_rank =
      fc1_tp_world_size(parallel_args) * kFc1LocalTokenAlignment;
  return geometry.min_local_num_tokens >= min_rows_per_cp_rank;
}

bool is_flash_comm1_eligible(int32_t num_tokens,
                             bool is_prefill,
                             const ParallelArgs& parallel_args,
                             const FlashComm1Options& options) {
  // No CP-local geometry available, so this caller shards over tp_group only.
  // Composing with CP here would shard the sequence twice.
  if (parallel_args.cp_size() != 1) {
    return false;
  }
  return is_flash_comm1_eligible(
      FlashComm1TokenGeometry::without_cp(num_tokens),
      is_prefill,
      parallel_args,
      options);
}

FlashComm1ContextScope::FlashComm1ContextScope(const FlashComm1Context* ctx)
    : previous_(current_flash_comm1_context) {
  current_flash_comm1_context = ctx;
}

FlashComm1ContextScope::~FlashComm1ContextScope() {
  if (previous_ == nullptr) {
    synchronize_quantized_mmrs_launches();
  }
  current_flash_comm1_context = previous_;
}

const FlashComm1Context* get_current_flash_comm1_context() {
  return current_flash_comm1_context;
}

void retain_quantized_mmrs_launch_tensors(const torch::Tensor& activation,
                                          const torch::Tensor& activation_scale,
                                          const torch::Tensor& weight_scale) {
  const std::array<torch::Tensor, 3> tensors = {
      activation, activation_scale, weight_scale};
  for (const torch::Tensor& tensor : tensors) {
    if (tensor.defined()) {
      quantized_mmrs_retained_bytes += tensor.nbytes();
      quantized_mmrs_launch_tensors.emplace_back(tensor);
    }
  }
  if (quantized_mmrs_retained_bytes >= kQuantizedMmrsRetainedBytesLimit) {
    synchronize_quantized_mmrs_launches();
  }
}

bool is_sequence_sharded(const FlashComm1Context& ctx) {
  return ctx.enabled && ctx.tp_world_size > 1;
}

torch::Tensor pad_rows_by_copy(const torch::Tensor& input,
                               int64_t padded_rows) {
  CHECK_GE(padded_rows, input.size(0));
  if (padded_rows == input.size(0)) {
    return input;
  }

  auto output_shape = input.sizes().vec();
  output_shape[0] = padded_rows;
  auto output = torch::empty(output_shape, input.options());
  output.slice(0, 0, input.size(0)).copy_(input);
  output.slice(0, input.size(0), padded_rows).zero_();
  return output;
}

FlashComm1Context build_flash_comm1_context(
    const FlashComm1TokenGeometry& geometry,
    bool is_prefill,
    const ParallelArgs& parallel_args,
    const FlashComm1Options& options) {
  FlashComm1Context ctx;

#if !defined(USE_NPU)
  return ctx;
#endif

  if (!is_flash_comm1_eligible(geometry, is_prefill, parallel_args, options)) {
    return ctx;
  }

  ProcessGroup* tp_group = parallel_args.tp_group_;
  if (!tp_group) {
    return ctx;
  }

  ctx.enabled = true;
  ctx.tp_rank = tp_group->rank();
  ctx.tp_world_size = tp_group->world_size();
  // Geometry follows the rows this rank actually holds: under an outer CP shard
  // that is the CP segment, not the pre-CP batch. shard_sequence() and the
  // row-parallel reduce paths all validate against original_num_tokens, so this
  // must be the local count.
  ctx.original_num_tokens = geometry.local_num_tokens;
  // MMRS fusion stays off under CP. The fused matmul+reduce_scatter kernel is
  // already shape-sensitive enough to be default-off (see enable_mmrs_fusion),
  // and an outer CP shard makes the row counts it sees both smaller and less
  // regular: local rows are a per-sequence CP segment rather than the whole
  // batch. The unfused reduce_scatter path is equivalent, so prefer it until
  // fused MMRS has been measured on CP-local shapes.
  const bool cp_active = std::max<int32_t>(parallel_args.cp_size(), 1) > 1;
  ctx.enable_mmrs_fusion = options.enable_mmrs_fusion && !cp_active;
  ctx.mmrs_comm_mode = options.mmrs_comm_mode;
  ctx.tp_group = tp_group;

  const int32_t token_alignment = ctx.tp_world_size * kFc1LocalTokenAlignment;
  ctx.padded_num_tokens =
      round_up_to_multiple(ctx.original_num_tokens, token_alignment);
  ctx.pad_size = ctx.padded_num_tokens - ctx.original_num_tokens;
  ctx.padded_local_num_tokens = ctx.padded_num_tokens / ctx.tp_world_size;

  return ctx;
}

FlashComm1Context build_flash_comm1_context(int32_t num_tokens,
                                            bool is_prefill,
                                            const ParallelArgs& parallel_args,
                                            const FlashComm1Options& options) {
  if (parallel_args.cp_size() != 1) {
    return FlashComm1Context{};
  }
  return build_flash_comm1_context(
      FlashComm1TokenGeometry::without_cp(num_tokens),
      is_prefill,
      parallel_args,
      options);
}

torch::Tensor shard_sequence(const torch::Tensor& input,
                             const FlashComm1Context& ctx) {
  if (!is_sequence_sharded(ctx)) {
    return input;
  }

  CHECK_EQ(input.size(0), ctx.original_num_tokens);
  const int64_t shard_start =
      static_cast<int64_t>(ctx.tp_rank) * ctx.padded_local_num_tokens;
  const int64_t shard_end = shard_start + ctx.padded_local_num_tokens;
  const int64_t valid_end =
      std::min(shard_end, static_cast<int64_t>(ctx.original_num_tokens));

  if (valid_end == shard_end) {
    return input.slice(0, shard_start, shard_end).contiguous();
  }

  auto output_shape = input.sizes().vec();
  output_shape[0] = ctx.padded_local_num_tokens;
  torch::Tensor output = torch::zeros(output_shape, input.options());
  if (valid_end > shard_start) {
    output.slice(0, 0, valid_end - shard_start)
        .copy_(input.slice(0, shard_start, valid_end));
  }
  return output;
}

torch::Tensor gather_sequence(const torch::Tensor& input,
                              const FlashComm1Context& ctx) {
  if (!is_sequence_sharded(ctx)) {
    return input;
  }

  const int32_t expected_local_size = ctx.padded_local_num_tokens;
  CHECK_EQ(input.size(0), expected_local_size)
      << "FC1 gather expects a padded local shard of " << expected_local_size
      << " rows, got " << input.size(0) << ", rank=" << ctx.tp_rank
      << ", world=" << ctx.tp_world_size;

  const std::vector<int32_t> token_nums(ctx.tp_world_size, expected_local_size);

  auto gathered = parallel_state::gather(input, ctx.tp_group, token_nums);

  if (ctx.pad_size > 0 && gathered.size(0) > ctx.original_num_tokens) {
    return gathered.slice(0, 0, ctx.original_num_tokens);
  }
  return gathered;
}

namespace {

torch::Tensor reduce_scatter_padded_local(const torch::Tensor& input,
                                          const FlashComm1Context& ctx) {
  CHECK(ctx.tp_group);
  CHECK(is_sequence_sharded(ctx));
  CHECK_EQ(input.size(0), ctx.original_num_tokens)
      << "FC1 row-parallel reduce_scatter expects full real-token output "
      << "before communication.";

  torch::Tensor padded_input = input;
  if (ctx.pad_size > 0) {
    padded_input = pad_rows_by_copy(input, ctx.padded_num_tokens);
  }

  auto output_shape = padded_input.sizes().vec();
  output_shape[0] = ctx.padded_local_num_tokens;
  torch::Tensor output = torch::empty(output_shape, padded_input.options());
  ctx.tp_group->reduce_scatter(padded_input, output);
  return output;
}

}  // namespace

torch::Tensor maybe_pad_and_reduce(torch::Tensor input,
                                   const FlashComm1Context& ctx,
                                   RowParallelReduceMode mode) {
  if (mode == RowParallelReduceMode::NONE) {
    return input;
  }

  CHECK(mode == RowParallelReduceMode::ALL_REDUCE ||
        mode == RowParallelReduceMode::REDUCE_SCATTER ||
        mode == RowParallelReduceMode::MATMUL_REDUCE_SCATTER)
      << "Unsupported row-parallel reduce mode.";

  if (!is_sequence_sharded(ctx)) {
    if (ctx.tp_group && ctx.tp_group->world_size() > 1) {
      return parallel_state::reduce(input, ctx.tp_group);
    }
    return input;
  }

  return reduce_scatter_padded_local(input, ctx);
}

RowParallelReduceMode row_parallel_reduce_mode_for_fc1(
    const FlashComm1Context& ctx) {
  return ctx.enable_mmrs_fusion ? RowParallelReduceMode::MATMUL_REDUCE_SCATTER
                                : RowParallelReduceMode::REDUCE_SCATTER;
}

torch::Tensor maybe_shard_residual(const torch::Tensor& residual,
                                   const FlashComm1Context& ctx) {
  if (!is_sequence_sharded(ctx)) {
    return residual;
  }
  const int64_t num_tokens = residual.size(0);
  CHECK(num_tokens == ctx.original_num_tokens ||
        num_tokens == ctx.padded_local_num_tokens)
      << "FC1 residual layout must be either full real-token or padded local "
      << "sequence shard.";
  if (num_tokens == ctx.original_num_tokens) {
    return shard_sequence(residual, ctx);
  }
  return residual;
}

}  // namespace xllm
