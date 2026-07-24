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

#include "framework/parallel_state/parallel_state.h"

namespace xllm {

namespace {

constexpr int32_t kFc1LocalTokenAlignment = 16;
constexpr int32_t kFc1MinTpSize = 8;
thread_local const FlashComm1Context* current_flash_comm1_context = nullptr;

int32_t round_up_to_multiple(int32_t value, int32_t multiple) {
  CHECK_GT(multiple, 0);
  const int32_t remainder = value % multiple;
  return remainder == 0 ? value : value + multiple - remainder;
}

}  // namespace

FlashComm1ContextScope::FlashComm1ContextScope(const FlashComm1Context* ctx)
    : previous_(current_flash_comm1_context) {
  current_flash_comm1_context = ctx;
}

FlashComm1ContextScope::~FlashComm1ContextScope() {
  current_flash_comm1_context = previous_;
}

const FlashComm1Context* get_current_flash_comm1_context() {
  return current_flash_comm1_context;
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

FlashComm1Context build_flash_comm1_context(int32_t num_tokens,
                                            bool is_prefill,
                                            const ParallelArgs& parallel_args,
                                            const FlashComm1Options& options) {
  int32_t actual_tp_size = parallel_args.world_size() /
                           (parallel_args.dp_size() * parallel_args.cp_size());

  FlashComm1Context ctx;

  if (!options.enable_flashcomm1) {
    return ctx;
  }

  if (!is_prefill) {
    return ctx;
  }

#if !defined(USE_NPU)
  return ctx;
#endif

  if (parallel_args.dp_size() != 1 || parallel_args.cp_size() != 1) {
    return ctx;
  }

  if (actual_tp_size < kFc1MinTpSize) {
    return ctx;
  }

  ProcessGroup* tp_group = parallel_args.tp_group_;
  if (!tp_group) {
    return ctx;
  }

  int32_t threshold = options.min_prefill_tokens;

  if (num_tokens < threshold) {
    return ctx;
  }

  ctx.enabled = true;
  ctx.tp_rank = tp_group->rank();
  ctx.tp_world_size = tp_group->world_size();
  ctx.original_num_tokens = num_tokens;
  ctx.enable_mmrs_fusion = options.enable_mmrs_fusion;
  ctx.mmrs_comm_mode = options.mmrs_comm_mode;
  ctx.tp_group = tp_group;

  const int32_t token_alignment = ctx.tp_world_size * kFc1LocalTokenAlignment;
  ctx.padded_num_tokens = round_up_to_multiple(num_tokens, token_alignment);
  ctx.pad_size = ctx.padded_num_tokens - num_tokens;
  ctx.padded_local_num_tokens = ctx.padded_num_tokens / ctx.tp_world_size;

  return ctx;
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
