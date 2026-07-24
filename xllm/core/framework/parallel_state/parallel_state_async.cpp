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

#include "core/util/utils.h"
#include "parallel_state.h"

namespace xllm {
namespace parallel_state {

GatherAsyncCtx launch_gather(const torch::Tensor& input,
                             ProcessGroup* process_group,
                             const std::vector<int32_t>& token_num_list) {
  GatherAsyncCtx ctx;
  if (!process_group) {
    ctx.input = input.contiguous();
    ctx.stacked = ctx.input.unsqueeze(0);
    ctx.token_num_list = {static_cast<int32_t>(input.size(0))};
    return ctx;
  }

  const int32_t world_size = process_group->world_size();
  const int32_t rank = process_group->rank();
  CHECK_EQ(token_num_list.size(), world_size)
      << "token_num_list size " << token_num_list.size()
      << " does not match world_size " << world_size;
  if (world_size == 1) {
    ctx.input = input.contiguous();
    ctx.stacked = ctx.input.unsqueeze(0);
    ctx.token_num_list = token_num_list;
    return ctx;
  }

  torch::Tensor contiguous_input = input.contiguous();
  CHECK_EQ(contiguous_input.size(0), token_num_list[rank])
      << "sequence-parallel gather local length mismatch: rank=" << rank
      << ", world_size=" << world_size
      << ", local_tensor_tokens=" << contiguous_input.size(0)
      << ", expected_tokens=" << token_num_list[rank];

  const int32_t max_num_tokens = xllm::util::max(token_num_list);
  const int32_t num_padding = max_num_tokens - token_num_list[rank];
  torch::Tensor padded_input = contiguous_input;
  if (num_padding > 0) {
    auto pad_shape = contiguous_input.sizes().vec();
    pad_shape[0] = num_padding;
    torch::Tensor pad_tensor =
        torch::zeros(pad_shape, contiguous_input.options());
    padded_input = torch::cat({contiguous_input, pad_tensor}, /*dim=*/0);
  }

  ctx.input = padded_input;
  auto stacked_shape = ctx.input.sizes().vec();
  stacked_shape.insert(stacked_shape.begin(), world_size);
  ctx.stacked = torch::empty(stacked_shape, ctx.input.options());
  ctx.token_num_list = token_num_list;
  ctx.work = process_group->allgather_base_async(ctx.input, ctx.stacked);
  return ctx;
}

ReduceAsyncCtx launch_reduce(torch::Tensor input, ProcessGroup* process_group) {
  ReduceAsyncCtx ctx;
  if (!process_group || process_group->world_size() <= 1) {
    ctx.tensor = std::move(input);
    return ctx;
  }

  ctx.tensor = input.contiguous();
  ctx.work = process_group->allreduce_async(ctx.tensor);
  return ctx;
}

AllGatherAsyncCtx launch_all_gather(const torch::Tensor& input,
                                    ProcessGroup* process_group) {
  AllGatherAsyncCtx ctx;
  if (!process_group || process_group->world_size() <= 1) {
    ctx.input = input.contiguous();
    ctx.stacked = ctx.input.unsqueeze(0);
    return ctx;
  }

  ctx.input = input.contiguous();
  auto stacked_shape = ctx.input.sizes().vec();
  stacked_shape.insert(stacked_shape.begin(), process_group->world_size());
  ctx.stacked = torch::empty(stacked_shape, ctx.input.options());
  ctx.work = process_group->allgather_base_async(ctx.input, ctx.stacked);
  return ctx;
}

torch::Tensor finish_all_gather(AllGatherAsyncCtx ctx) {
  if (ctx.work.defined()) {
    ctx.work->wait();
  }
  return ctx.stacked;
}

AllToAllAsyncCtx launch_all_to_all(const torch::Tensor& input,
                                   ProcessGroup* process_group) {
  AllToAllAsyncCtx ctx;
  if (!process_group || process_group->world_size() <= 1) {
    ctx.input = input.contiguous();
    ctx.output = ctx.input;
    return ctx;
  }

  const int32_t world_size = process_group->world_size();
  ctx.input = input.contiguous();
  CHECK_EQ(ctx.input.size(0) % world_size, 0)
      << "launch_all_to_all: input dim 0 (" << ctx.input.size(0)
      << ") must be divisible by world_size (" << world_size << ")";
  ctx.output = torch::empty_like(ctx.input);
  process_group->all_to_all_single(ctx.output,
                                   ctx.input,
                                   /*output_split_sizes=*/{},
                                   /*input_split_sizes=*/{},
                                   /*async_op=*/true,
                                   &ctx.work);
  return ctx;
}

torch::Tensor finish_all_to_all(AllToAllAsyncCtx ctx) {
  if (ctx.work.defined()) {
    ctx.work->wait();
  }
  return ctx.output;
}
}  // namespace parallel_state
}  // namespace xllm
