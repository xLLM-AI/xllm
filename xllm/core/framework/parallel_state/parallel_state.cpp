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

#include "parallel_state.h"

#include "core/util/utils.h"
#include "runtime/options.h"
#include "util/net.h"

#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUFunctions.h>

#include "framework/parallel_state/single_process/hccl_process_group.h"
#include "hccl/hccl.h"
#include "npu_process_group.h"
#endif

namespace xllm {
namespace parallel_state {

namespace {

torch::Tensor assemble_gathered(const torch::Tensor& stacked_tensor,
                                const std::vector<int32_t>& token_num_list) {
  CHECK(stacked_tensor.defined()) << "stacked_tensor must be defined.";
  CHECK_GT(stacked_tensor.dim(), 1)
      << "stacked_tensor must be stacked by rank.";
  CHECK_EQ(stacked_tensor.size(0), static_cast<int64_t>(token_num_list.size()))
      << "stacked_tensor size " << stacked_tensor.size(0)
      << " does not match token_num_list size " << token_num_list.size();

  int64_t total_tokens = xllm::util::sum(token_num_list);
  auto out_shape = stacked_tensor[0].sizes().vec();
  out_shape[0] = total_tokens;
  torch::Tensor output = torch::empty(out_shape, stacked_tensor.options());

  int64_t offset = 0;
  for (size_t i = 0; i < token_num_list.size(); ++i) {
    const int32_t valid_tokens = token_num_list[i];
    if (valid_tokens <= 0) {
      continue;
    }
    CHECK_GE(stacked_tensor[static_cast<int64_t>(i)].size(0), valid_tokens)
        << "sequence-parallel gather received fewer rows than expected: "
        << "src_rank=" << i
        << ", gathered_rows=" << stacked_tensor[static_cast<int64_t>(i)].size(0)
        << ", expected_rows=" << valid_tokens;
    output.slice(0, offset, offset + valid_tokens)
        .copy_(
            stacked_tensor[static_cast<int64_t>(i)].slice(0, 0, valid_tokens));
    offset += valid_tokens;
  }
  return output;
}

}  // namespace

std::optional<ParallelArgs> get_dp_attn_parallel_args(
    const ParallelArgs& parallel_args) {
  if (parallel_args.dp_size() <= 1) {
    return std::nullopt;
  }

  // tp=1 in each dp group
  if (parallel_args.dp_size() == parallel_args.world_size()) {
    return ParallelArgs(0,  // local rank
                        1,  // world_size
                        nullptr,
                        nullptr,
                        parallel_args.dp_size());
  }

  return ParallelArgs(parallel_args.dp_local_process_group_->rank(),
                      parallel_args.dp_local_process_group_->world_size(),
                      parallel_args.dp_local_process_group_,
                      nullptr,
                      parallel_args.dp_size());
}

torch::Tensor gather(const torch::Tensor& input,
                     ProcessGroup* process_group,
                     int32_t dim) {
  if (!process_group) {
    return input;
  }
  const int32_t world_size = process_group->world_size();
  if (world_size == 1) {
    return input;
  }

  torch::Tensor stacked = process_group->allgather_base_sync(input);
  return torch::cat(stacked.unbind(0), /*dim=*/dim).contiguous();
}

torch::Tensor gather(const torch::Tensor& input,
                     ProcessGroup* process_group,
                     const std::vector<int32_t>& token_num_list) {
  if (!process_group) return input;
  const int32_t world_size = process_group->world_size();
  if (world_size == 1) return input;

  CHECK_EQ(token_num_list.size(), world_size)
      << "token_num_list size " << token_num_list.size()
      << " does not match world_size " << world_size;

  const bool num_tokens_equal =
      std::all_of(token_num_list.begin(),
                  token_num_list.end(),
                  [first_token_num = token_num_list[0]](int64_t num) {
                    return num == first_token_num;
                  });
  if (num_tokens_equal) {
    return gather(input, process_group, 0);
  }
  return finish_gather(launch_gather(input, process_group, token_num_list));
}

torch::Tensor finish_gather(GatherAsyncCtx ctx) {
  if (ctx.work.defined()) {
    ctx.work->wait();
  }
  if (ctx.stacked.defined() && ctx.stacked.size(0) == 1 &&
      ctx.token_num_list.size() == 1) {
    if (ctx.stacked[0].size(0) == ctx.token_num_list.front()) {
      return ctx.stacked[0];
    }
  }
  const bool num_tokens_equal =
      !ctx.token_num_list.empty() &&
      std::all_of(ctx.token_num_list.begin(),
                  ctx.token_num_list.end(),
                  [first_token_num = ctx.token_num_list[0]](int64_t num) {
                    return num == first_token_num;
                  });
  if (num_tokens_equal) {
    return ctx.stacked.flatten(0, 1).contiguous();
  }
  return assemble_gathered(ctx.stacked, ctx.token_num_list);
}

torch::Tensor all_gather_interleaved(const torch::Tensor& input,
                                     ProcessGroup* process_group) {
  if (!process_group) {
    return input;
  }
  const int32_t world_size = process_group->world_size();
  if (world_size == 1) {
    return input;
  }

  torch::Tensor gathered_tensors = process_group->allgather_base_sync(input);

  int32_t dim = -1;
  size_t num_chunks = 3;
  std::vector<torch::Tensor> ordered_tensors;
  int64_t shard_size = input.size(dim) / num_chunks;
  for (size_t i = 0; i < num_chunks; ++i) {
    for (size_t j = 0; j < world_size; ++j) {
      auto shard_tensor =
          gathered_tensors[j].slice(dim, shard_size * i, shard_size * (i + 1));
      ordered_tensors.push_back(shard_tensor);
    }
  }
  return torch::cat(ordered_tensors, dim).contiguous();
}

torch::Tensor finish_reduce(ReduceAsyncCtx ctx) {
  if (ctx.work.defined()) {
    ctx.work->wait();
  }
  return ctx.tensor;
}

torch::Tensor reduce(torch::Tensor& input, ProcessGroup* process_group) {
  if (!process_group) {
    return input;
  }
  const int32_t world_size = process_group->world_size();
  if (world_size == 1) {
    return input;
  }
  return finish_reduce(launch_reduce(input, process_group));
}

torch::Tensor reduce_scatter(const torch::Tensor& input,
                             ProcessGroup* process_group) {
  // currently only support scatter_dim == 0
  if (!process_group) return input;
  const int32_t world_size = process_group->world_size();
  if (world_size == 1) return input;

  const int32_t rank = process_group->rank();
  const int64_t original_dim_size = input.size(0);

  // check if padding is needed
  // round up to the nearest multiple of world_size: (N + W - 1) / W * W or N +
  // (W - N%W)%W
  int64_t remainder = original_dim_size % world_size;
  int64_t target_size = (remainder == 0)
                            ? original_dim_size
                            : (original_dim_size + world_size - remainder);
  int64_t num_padding = target_size - original_dim_size;
  torch::Tensor padded_input = input;
  if (num_padding > 0) {
    std::vector<int64_t> pad = {0, 0, 0, num_padding};
    // Explicitly calling kConstant and value of 0 ensures consistency across
    // platforms and versions.
    padded_input = torch::nn::functional::pad(
        input, torch::nn::functional::PadFuncOptions(pad));
  }

  // prepare output tensor
  // at this point, padded_input size along dim 0 is divisible by world_size
  const int64_t padded_dim_size = padded_input.size(0);
  const int64_t chunk_size = padded_dim_size / world_size;

  auto output_shape = padded_input.sizes().vec();
  output_shape[0] = chunk_size;
  torch::Tensor output = torch::empty(output_shape, padded_input.options());

  // perform reduce scatter operation
  process_group->reduce_scatter(padded_input, output);

  // remove padding
  if (num_padding > 0) {
    int64_t global_start = rank * chunk_size;
    int64_t global_end = global_start + chunk_size;

    if (global_start >= original_dim_size) {
      return output.slice(0, 0, 0);
    } else if (global_end > original_dim_size) {
      return output.slice(0, 0, original_dim_size - global_start);
    }
  }

  return output;
}

torch::Tensor scatter(torch::Tensor input,
                      ProcessGroup* process_group,
                      int dim) {
  if (!process_group) {
    return input;
  }
  const int32_t world_size = process_group->world_size();
  if (world_size == 1) {
    return input;
  }

  // get the size for last dimension
  const int32_t dim_size = input.size(dim);
  CHECK(dim_size % world_size == 0)
      << "dim_size " << dim_size << " cannot be divided by world_size "
      << world_size;

  // torch::split does not create contiguous tensors by default.
  const auto tensor_list = input.split(dim_size / world_size, dim);
  const int32_t rank = process_group->rank();
  return tensor_list[rank];
}

std::function<torch::Tensor()> all_to_all_4D(const torch::Tensor& input,
                                             int32_t scatter_idx,
                                             int32_t gather_idx,
                                             bool async_ops,
                                             ProcessGroup* process_group) {
  if (!process_group) {
    return [input]() { return input; };
  }
  const int32_t group_size = process_group->world_size();

  if (group_size == 1) {
    return [input]() { return input; };
  }

  auto rank = process_group->rank();

  TORCH_CHECK(input.dim() == 4,
              "all_to_all_4D: input must be 4D, got dim=",
              input.dim());
  auto send_input = input;

  if (scatter_idx == 2 && gather_idx == 1) {
    // branch A : from "sequence shard" -> "head shard"
    // input: (bs, seqlen / group_size (shard_seqlen), head_num, head_dim)
    //   output (bs, seqlen, head_num / group_size, head_dim)
    auto sizes = send_input.sizes().vec();
    const int64_t bs = sizes[0];
    const int64_t shard_seqlen = sizes[1];
    const int64_t head_num = sizes[2];
    const int64_t head_size = sizes[3];
    const int64_t seqlen = shard_seqlen * group_size;
    TORCH_CHECK(head_num % group_size == 0,
                "all_to_all_4D(A): head_num must be divisible by group_size");
    const int64_t shard_head_num = head_num / group_size;

    // prepare expected shape for All2All (group_size, shard_seqlen, bs,
    // shard_head_num, head_size)
    auto input_t =
        send_input
            .reshape({bs, shard_seqlen, group_size, shard_head_num, head_size})
            .transpose(
                0,
                2)  // (group_size, shard_seqlen, bs, shard_head_num, head_size)
            .contiguous();
    torch::Tensor output = torch::empty_like(input_t);
    std::vector<int64_t> input_split_size = {};
    std::vector<int64_t> output_split_size = {};

    if (!async_ops) {
      process_group->all_to_all_single(
          output, input_t, output_split_size, input_split_size, async_ops);
      output = output.reshape({seqlen, bs, shard_head_num, head_size})
                   .transpose(0, 1)
                   .contiguous()
                   .reshape({bs, seqlen, shard_head_num, head_size});
      return [output]() { return output; };
    } else {
      c10::intrusive_ptr<c10d::Work> all2all_work;
      process_group->all_to_all_single(output,
                                       input_t,
                                       output_split_size,
                                       input_split_size,
                                       async_ops,
                                       &all2all_work);
      return [output,
              all2all_work,
              bs,
              seqlen,
              shard_head_num,
              head_size]() mutable -> torch::Tensor {
        all2all_work->wait();
        auto comm_output =
            output.reshape({seqlen, bs, shard_head_num, head_size})
                .transpose(0, 1)
                .contiguous()
                .reshape({bs, seqlen, shard_head_num, head_size});
        return comm_output;
      };
    }
  } else if (scatter_idx == 1 && gather_idx == 2) {
    // branch B : from "head shard" -> "sequence shard"
    // input: (bs, seqlen, head_num / group_size, head_size)
    // output (bs, seqlen / group_size, head_num, haed_size)
    auto sizes = send_input.sizes().vec();
    const int64_t bs = sizes[0];
    const int64_t seqlen = sizes[1];
    const int64_t shard_head_num = sizes[2];
    const int64_t head_size = sizes[3];
    TORCH_CHECK(seqlen % group_size == 0,
                "all_to_all_4D(B): seqlen must be divisible by group_size");
    const int64_t shard_seqlen = seqlen / group_size;
    const int64_t head_num = shard_head_num * group_size;

    // prepare expected shape for All2All (group_size, shard_head_num,
    // shard_seqlen, bs, head_size)
    auto input_t =
        send_input
            .reshape({bs, group_size, shard_seqlen, shard_head_num, head_size})
            .transpose(
                0,
                3)  // (shard_head_num, group_size, shard_seqlen, bs, head_size)
            .transpose(
                0,
                1)  // (group_size, shard_head_num, shard_seqlen, bs, head_size)
            .contiguous();
    torch::Tensor output = torch::empty_like(input_t);
    std::vector<int64_t> input_split_size = {};
    std::vector<int64_t> output_split_size = {};

    if (!async_ops) {
      process_group->all_to_all_single(output,
                                       input_t,
                                       output_split_size,
                                       input_split_size,
                                       /*async_op=*/false);
      output = output.reshape({head_num, shard_seqlen, bs, head_size})
                   .transpose(0, 2)
                   .contiguous()
                   .reshape({bs, shard_seqlen, head_num, head_size});
      return [output]() { return output; };
    } else {
      c10::intrusive_ptr<c10d::Work> all2all_work;
      process_group->all_to_all_single(output,
                                       input_t,
                                       output_split_size,
                                       input_split_size,
                                       /*async_op=*/true,
                                       &all2all_work);
      return [output,
              all2all_work,
              head_num,
              shard_seqlen,
              bs,
              head_size]() mutable -> torch::Tensor {
        all2all_work->wait();
        auto comm_output =
            output.reshape({head_num, shard_seqlen, bs, head_size})
                .transpose(0, 2)
                .contiguous()
                .reshape({bs, shard_seqlen, head_num, head_size});
        return comm_output;
      };
    }
  } else {
    TORCH_CHECK(false,
                "all_to_all_4D: only (scatter_idx,gather_idx)=(2,1) or (1,2) "
                "are supported");
  }
}

std::vector<std::unique_ptr<ProcessGroup>> create_npu_process_groups(
    const std::vector<torch::Device>& devices) {
#if defined(USE_NPU)
  CHECK(!devices.empty()) << "devices should not be empty";

  std::vector<int> device_idxs;
  device_idxs.reserve(devices.size());
  for (const auto& device : devices) {
    device_idxs.push_back(device.index());
  }

  std::vector<HcclComm> comms(devices.size());
  const int32_t world_size = static_cast<int32_t>(devices.size());
  HCCLCHECK(HcclCommInitAll(world_size, device_idxs.data(), comms.data()));

  std::vector<std::unique_ptr<ProcessGroup>> process_groups;
  process_groups.reserve(devices.size());
  for (int32_t i = 0; i < world_size; ++i) {
    process_groups.emplace_back(std::make_unique<ProcessGroupImpl>(
        /*rank=*/i, world_size, devices[i], comms[i]));
  }

  return process_groups;
#else
  LOG(FATAL) << "non-NPU device is not supported";
#endif
}

std::vector<std::unique_ptr<ProcessGroup>> create_local_process_groups(
    const std::vector<torch::Device>& devices,
    const runtime::Options& options) {
  CHECK(!devices.empty()) << "devices should not be empty";
  const int32_t world_size = static_cast<int32_t>(devices.size());

  std::vector<std::unique_ptr<ProcessGroup>> process_groups;
  process_groups.reserve(devices.size());

#if defined(USE_NPU)
  // Single-process multi-device path: use HcclCommInitAll so every device
  // shares one HCCL world. Two c10d_npu::ProcessGroupHCCL instances coexisting
  // inside the same process can dispatch collectives in mismatched orders and
  // deadlock the local_tp_group under load; HcclCommInitAll-backed groups
  // sidestep that by sharing a single HCCL bootstrap.
  std::vector<int> device_idxs;
  device_idxs.reserve(devices.size());
  for (const auto& device : devices) {
    device_idxs.push_back(device.index());
  }
  std::vector<HcclComm> comms(devices.size());
  // HcclCommInitAll requires a device bound on the calling thread to refresh
  // the runtime device logic id; the engine thread that runs this has not
  // bound one yet (workers bind their own device on their own threads later).
  // Bind through c10_npu::SetDevice rather than raw aclrtSetDevice so
  // torch_npu's thread-local device cache stays in sync; a raw aclrtSetDevice
  // would desync that cache and make later c10_npu::SetDevice calls skip the
  // real device switch, leaving worker streams in the wrong context.
  int32_t prev_device = c10_npu::current_device();
  CHECK_EQ(c10_npu::SetDevice(static_cast<int32_t>(device_idxs[0])),
           ACL_SUCCESS)
      << "c10_npu::SetDevice failed for device " << device_idxs[0]
      << " before HcclCommInitAll";
  HCCLCHECK(HcclCommInitAll(world_size, device_idxs.data(), comms.data()));
  if (prev_device >= 0) {
    CHECK_EQ(c10_npu::SetDevice(prev_device), ACL_SUCCESS)
        << "c10_npu::SetDevice failed to restore device " << prev_device;
  }
  for (int32_t i = 0; i < world_size; ++i) {
    process_groups.emplace_back(std::make_unique<HcclProcessGroup>(
        /*rank=*/i, world_size, devices[i], comms[i]));
  }
#elif defined(USE_CUDA) || defined(USE_MLU) || defined(USE_ILU) || \
    defined(USE_DCU)
  // For GPU: use create_process_group with localhost
  // Parse port from options.master_node_addr() to support multiple instances
  std::string host;
  int port;

  // Parse port from options.master_node_addr()
  // Note: master_node_addr always has a default value (127.0.0.1:19888)
  net::parse_host_port_from_addr(
      options.master_node_addr().value(), host, port);

  // Override host to localhost for local communication
  host = "127.0.0.1";

  for (int32_t i = 0; i < world_size; ++i) {
    process_groups.emplace_back(create_process_group(
        /*rank=*/i,
        /*world_size=*/world_size,
        /*rank_size=*/world_size,
        /*port=*/port,
        /*trans=*/false,
        host,
        /*group_name=*/"local_tp_group",
        devices[i]));
  }
#else
  LOG(FATAL) << "Unsupported device type for create_local_process_groups";
#endif

  return process_groups;
}

}  // namespace parallel_state
}  // namespace xllm
