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

#include "common/global_flags.h"
#include "framework/parallel_state/parallel_state.h"
#include "platform/device.h"

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
            true,
            "Enable Matmul+ReduceScatter fusion kernel for FC1.");

namespace xllm {

FlashComm1Context build_flash_comm1_context(int32_t num_tokens,
                                            bool is_prefill,
                                            const ParallelArgs& parallel_args) {
  int32_t actual_tp_size = parallel_args.world_size() /
                           (parallel_args.dp_size() * parallel_args.cp_size());

  FlashComm1Context ctx;

  if (!FLAGS_enable_flashcomm1) {
    return ctx;
  }

  if (Device::type_str() != "npu") {
    return ctx;
  }

  if (actual_tp_size <= 1) {
    return ctx;
  }

  ProcessGroup* tp_group = parallel_args.tp_group_;
  if (!tp_group) {
    return ctx;
  }

  int32_t threshold = is_prefill ? FLAGS_flashcomm1_min_prefill_tokens
                                 : FLAGS_flashcomm1_min_decode_tokens;

  if (num_tokens < threshold) {
    return ctx;
  }

  ctx.enabled = true;
  ctx.tp_rank = tp_group->rank();
  ctx.tp_world_size = tp_group->world_size();
  ctx.original_num_tokens = num_tokens;
  ctx.is_prefill = is_prefill;
  ctx.tp_group = tp_group;

  int32_t remainder = num_tokens % ctx.tp_world_size;
  ctx.pad_size = remainder == 0 ? 0 : ctx.tp_world_size - remainder;
  ctx.padded_num_tokens = num_tokens + ctx.pad_size;
  ctx.local_num_tokens = ctx.padded_num_tokens / ctx.tp_world_size;

  return ctx;
}

torch::Tensor shard_sequence(const torch::Tensor& input,
                             const FlashComm1Context& ctx) {
  if (!ctx.is_sequence_sharded()) {
    return input;
  }

  auto chunks = input.chunk(ctx.tp_world_size, 0);
  return chunks[ctx.tp_rank].contiguous();
}

torch::Tensor gather_sequence(const torch::Tensor& input,
                              const FlashComm1Context& ctx) {
  if (!ctx.is_sequence_sharded()) {
    return input;
  }

  int32_t current_local_size = input.size(0);
  int32_t expected_even_size = ctx.padded_num_tokens / ctx.tp_world_size;
  int32_t expected_uneven_base = ctx.original_num_tokens / ctx.tp_world_size;
  int32_t expected_uneven_remainder =
      ctx.original_num_tokens % ctx.tp_world_size;
  int32_t expected_uneven_for_rank =
      expected_uneven_base + (ctx.tp_rank < expected_uneven_remainder ? 1 : 0);

  std::vector<int32_t> token_num_list(ctx.tp_world_size);
  if (current_local_size == expected_even_size && ctx.pad_size > 0) {
    for (int32_t i = 0; i < ctx.tp_world_size; ++i) {
      token_num_list[i] = expected_even_size;
    }
  } else {
    int32_t base_size = ctx.original_num_tokens / ctx.tp_world_size;
    int32_t remainder = ctx.original_num_tokens % ctx.tp_world_size;
    for (int32_t i = 0; i < ctx.tp_world_size; ++i) {
      token_num_list[i] = base_size + (i < remainder ? 1 : 0);
    }
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

  auto options = input.options();
  auto padding = torch::zeros({dynamic_pad_size, input.size(-1)}, options);
  return torch::cat({input, padding}, 0);
}

torch::Tensor maybe_pad_and_reduce(torch::Tensor input,
                                   const FlashComm1Context& ctx,
                                   RowParallelReduceMode mode) {
  if (mode == RowParallelReduceMode::NONE) {
    return input;
  }

  if (!ctx.is_sequence_sharded()) {
    if (ctx.tp_group && ctx.tp_group->world_size() > 1) {
      return parallel_state::reduce(input, ctx.tp_group);
    }
    return input;
  }

  auto padded = maybe_pad_for_reduce(input, ctx);
  return parallel_state::reduce_scatter(padded, ctx.tp_group);
}

torch::Tensor maybe_chunk_residual(const torch::Tensor& residual,
                                   int32_t tp_rank,
                                   int32_t tp_world_size) {
  if (tp_world_size <= 1) {
    return residual;
  }

  auto chunks = residual.chunk(tp_world_size, 0);
  return chunks[tp_rank].contiguous();
}

}  // namespace xllm
