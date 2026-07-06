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

#include "common/global_flags.h"
#include "framework/parallel_state/parallel_state.h"
#if defined(USE_NPU)
#include "platform/device.h"
#endif

DEFINE_bool(enable_flashcomm1,
            false,
            "Enable Flash Communication 1 (FC1) sequence-parallel optimization "
            "for tensor parallel inference on NPU.");

DEFINE_int32(flashcomm1_min_prefill_tokens,
             1000,
             "Minimum prefill token count to activate FC1.");

DEFINE_int32(flashcomm1_min_decode_tokens,
             128,
             "Minimum decode batch token count to activate FC1.");

DEFINE_bool(enable_mmrs_fusion,
            false,
            "Enable Matmul+ReduceScatter fusion kernel for FC1.");

DEFINE_string(mmrs_comm_mode,
              "aiv",
              "Communication mode for torch_npu npu_mm_reduce_scatter_base. "
              "Supported values: ai_cpu, aiv, none.");

namespace xllm {

namespace {

constexpr int32_t kFc1LocalTokenAlignment = 16;

int32_t round_up_to_multiple(int32_t value, int32_t multiple) {
  CHECK_GT(multiple, 0);
  const int32_t remainder = value % multiple;
  return remainder == 0 ? value : value + multiple - remainder;
}

int32_t local_num_tokens_for_rank(int32_t num_tokens,
                                  int32_t world_size,
                                  int32_t rank) {
  const int32_t base = num_tokens / world_size;
  const int32_t remainder = num_tokens % world_size;
  return base + (rank < remainder ? 1 : 0);
}

int64_t shard_start_for_rank(int32_t num_tokens,
                             int32_t world_size,
                             int32_t rank) {
  const int32_t base = num_tokens / world_size;
  const int32_t remainder = num_tokens % world_size;
  return static_cast<int64_t>(rank) * base + std::min(rank, remainder);
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

}  // namespace

int32_t FlashComm1Context::local_num_tokens_for_rank(int32_t rank) const {
  CHECK_GE(rank, 0);
  CHECK_LT(rank, tp_world_size);
  return xllm::local_num_tokens_for_rank(
      original_num_tokens, tp_world_size, rank);
}

std::vector<int32_t> FlashComm1Context::token_num_list() const {
  std::vector<int32_t> token_nums(tp_world_size);
  for (int32_t rank = 0; rank < tp_world_size; ++rank) {
    token_nums[rank] = local_num_tokens_for_rank(rank);
  }
  return token_nums;
}

int64_t FlashComm1Context::get_shard_start() const {
  return shard_start_for_rank(original_num_tokens, tp_world_size, tp_rank);
}

int64_t FlashComm1Context::get_shard_end() const {
  return get_shard_start() + local_num_tokens;
}

FlashComm1Context build_flash_comm1_context(
    int32_t num_tokens,
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

#if defined(USE_NPU)
  if (Device::type_str() != "npu") {
    return ctx;
  }
#else
  return ctx;
#endif

  if (parallel_args.dp_size() != 1 || parallel_args.cp_size() != 1) {
    return ctx;
  }

  if (actual_tp_size <= 1) {
    return ctx;
  }

  ProcessGroup* tp_group = parallel_args.tp_group_;
  if (!tp_group) {
    return ctx;
  }

  int32_t threshold = options.min_prefill_tokens;

  if (num_tokens <= threshold) {
    return ctx;
  }

  ctx.enabled = true;
  ctx.tp_rank = tp_group->rank();
  ctx.tp_world_size = tp_group->world_size();
  ctx.original_num_tokens = num_tokens;
  ctx.is_prefill = is_prefill;
  ctx.enable_mmrs_fusion = options.enable_mmrs_fusion;
  ctx.mmrs_comm_mode = options.mmrs_comm_mode;
  ctx.tp_group = tp_group;

  const int32_t token_alignment =
      ctx.tp_world_size * kFc1LocalTokenAlignment;
  ctx.padded_num_tokens = round_up_to_multiple(num_tokens, token_alignment);
  ctx.pad_size = ctx.padded_num_tokens - num_tokens;
  ctx.local_num_tokens = ctx.local_num_tokens_for_rank(ctx.tp_rank);
  ctx.padded_local_num_tokens = ctx.padded_num_tokens / ctx.tp_world_size;

  return ctx;
}

FlashComm1Context build_flash_comm1_context(int32_t num_tokens,
                                            bool is_prefill,
                                            const ParallelArgs& parallel_args) {
  FlashComm1Options options;
  options.enable_flashcomm1 = FLAGS_enable_flashcomm1;
  options.min_prefill_tokens = FLAGS_flashcomm1_min_prefill_tokens;
  options.min_decode_tokens = FLAGS_flashcomm1_min_decode_tokens;
  options.enable_mmrs_fusion = FLAGS_enable_mmrs_fusion;
  options.mmrs_comm_mode = FLAGS_mmrs_comm_mode;
  return build_flash_comm1_context(
      num_tokens, is_prefill, parallel_args, options);
}

torch::Tensor shard_sequence(const torch::Tensor& input,
                             const FlashComm1Context& ctx) {
  if (!ctx.is_sequence_sharded()) {
    return input;
  }

  CHECK_EQ(input.size(0), ctx.original_num_tokens);
  torch::Tensor padded_input = input;
  if (ctx.pad_size > 0) {
    padded_input = pad_rows_by_copy(input, ctx.padded_num_tokens);
  }

  const int64_t shard_start =
      static_cast<int64_t>(ctx.tp_rank) * ctx.padded_local_num_tokens;
  const int64_t shard_end = shard_start + ctx.padded_local_num_tokens;
  return padded_input.slice(0, shard_start, shard_end).contiguous();
}

torch::Tensor gather_sequence(const torch::Tensor& input,
                              const FlashComm1Context& ctx) {
  if (!ctx.is_sequence_sharded()) {
    return input;
  }

  const int32_t current_local_size = input.size(0);
  const int32_t expected_even_size = ctx.padded_num_tokens / ctx.tp_world_size;
  const int32_t expected_local_size = ctx.local_num_tokens;

  std::vector<int32_t> token_num_list = ctx.token_num_list();
  if (ctx.pad_size > 0 && current_local_size == expected_even_size) {
    for (int32_t i = 0; i < ctx.tp_world_size; ++i) {
      token_num_list[i] = expected_even_size;
    }
  } else {
    CHECK_EQ(current_local_size, expected_local_size)
        << "FC1 gather expected local real-token shard size "
        << expected_local_size << ", got " << current_local_size
        << ", rank=" << ctx.tp_rank << ", world=" << ctx.tp_world_size;
  }

  auto gathered = parallel_state::gather(input, ctx.tp_group, token_num_list);

  if (ctx.pad_size > 0 && gathered.size(0) > ctx.original_num_tokens) {
    return gathered.slice(0, 0, ctx.original_num_tokens);
  }
  return gathered;
}

torch::Tensor gather_and_unpad_sequence(const torch::Tensor& input,
                                        const FlashComm1Context& ctx) {
  auto gathered = gather_sequence(input, ctx);
  if (ctx.pad_size > 0) {
    return gathered.slice(0, 0, ctx.original_num_tokens);
  }
  return gathered;
}

torch::Tensor maybe_pad_for_reduce(const torch::Tensor& input,
                                   const FlashComm1Context& ctx) {
  int32_t current_size = input.size(0);
  int32_t remainder = current_size % ctx.tp_world_size;

  if (remainder == 0) {
    return input;
  }

  int32_t dynamic_pad_size = ctx.tp_world_size - remainder;
  return pad_rows_by_copy(input, current_size + dynamic_pad_size);
}

namespace {

torch::Tensor reduce_scatter_padded_local(const torch::Tensor& input,
                                          const FlashComm1Context& ctx) {
  CHECK(ctx.tp_group);
  CHECK(ctx.is_sequence_sharded());
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
  LOG_FIRST_N(INFO, 16)
      << "FC1 reduce_scatter active: rank=" << ctx.tp_rank
      << ", original_tokens=" << ctx.original_num_tokens
      << ", padded_tokens=" << ctx.padded_num_tokens
      << ", local_tokens=" << ctx.padded_local_num_tokens;
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

  if (!ctx.is_sequence_sharded()) {
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

torch::Tensor maybe_chunk_residual(const torch::Tensor& residual,
                                   int32_t tp_rank,
                                   int32_t tp_world_size) {
  if (tp_world_size <= 1) {
    return residual;
  }

  CHECK_GE(tp_rank, 0);
  CHECK_LT(tp_rank, tp_world_size);
  const int32_t num_tokens = static_cast<int32_t>(residual.size(0));
  const int64_t start =
      shard_start_for_rank(num_tokens, tp_world_size, tp_rank);
  const int64_t end =
      start + local_num_tokens_for_rank(num_tokens, tp_world_size, tp_rank);
  return residual.slice(0, start, end).contiguous();
}

torch::Tensor maybe_shard_residual(const torch::Tensor& residual,
                                   const FlashComm1Context& ctx) {
  if (!ctx.is_sequence_sharded()) {
    return residual;
  }
  if (residual.size(0) == ctx.padded_local_num_tokens) {
    return residual;
  }
  if (residual.size(0) == ctx.original_num_tokens) {
    return shard_sequence(residual, ctx);
  }
  CHECK_EQ(residual.size(0), ctx.padded_local_num_tokens)
      << "FC1 residual layout must be either full real-token or padded local "
      << "sequence shard.";
  return residual;
}

}  // namespace xllm
