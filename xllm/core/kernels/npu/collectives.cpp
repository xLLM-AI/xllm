/* Copyright 2026 The xLLM Authors.

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

// Current-stream HCCL collectives submitted directly on the NPU stream that is
// current at the call site. The caller owns the communicator (passed as an
// integer handle for Torch), the buffers, and their lifetime; nothing is
// created or cached here. Because each primitive joins the caller's stream
// instead of forking to a private one, a capture records it in the captured
// stream order and replay re-issues it without a wait/record pair.
//
// ACLGraph capture of these primitives requires HCCL_OP_EXPANSION_MODE=AIV;
// the AICPU expansion mode cannot be captured.

#include <ATen/MemoryOverlap.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/aten/CustomFunctions.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#endif
#include <hccl/hccl.h>

#include "npu_ops_api.h"

namespace xllm::kernel::npu {
namespace {

bool is_npu(const torch::Tensor& tensor) {
  return tensor.defined() && tensor.device().is_privateuseone();
}

HcclDataType to_hccl_data_type(const torch::Tensor& input) {
  const torch::ScalarType type = input.scalar_type();
  switch (type) {
    case torch::kFloat:
      return HCCL_DATA_TYPE_FP32;
    case torch::kHalf:
      return HCCL_DATA_TYPE_FP16;
    case torch::kDouble:
      return HCCL_DATA_TYPE_FP64;
    case torch::kLong:
      return HCCL_DATA_TYPE_INT64;
    case torch::kInt:
      return HCCL_DATA_TYPE_INT32;
    case torch::kChar:
      return HCCL_DATA_TYPE_INT8;
    case torch::kByte:
      return HCCL_DATA_TYPE_UINT8;
    case torch::kBool:
      return HCCL_DATA_TYPE_UINT8;
    case torch::kBFloat16:
      return HCCL_DATA_TYPE_BFP16;
    default:
      LOG(FATAL) << "Unconvertible HCCL type " << type;
      return HCCL_DATA_TYPE_FP32;
  }
}

void check_input(const torch::Tensor& input) {
  CHECK(is_npu(input)) << "HCCL requires an NPU tensor.";
  CHECK(input.layout() == torch::kStrided) << "HCCL requires a dense tensor.";
  CHECK(input.is_contiguous()) << "HCCL requires a contiguous tensor.";
  CHECK_GT(input.numel(), 0) << "HCCL requires a nonempty tensor.";
#ifdef TORCH_HIGHER_THAN_PTA6
  const int64_t format = at_npu::native::get_npu_format(input);
#else
  const int64_t format =
      at_npu::native::NPUNativeFunctions::get_npu_format(input);
#endif
  CHECK_EQ(format, ACL_FORMAT_ND) << "HCCL requires ND storage format.";
}

c10_npu::NPUStream collective_stream(const torch::Tensor& input, int64_t comm) {
  check_input(input);
  CHECK_NE(comm, 0) << "HCCL communicator must not be null.";
  const auto stream = c10_npu::getCurrentNPUStream();
  CHECK_EQ(stream.device_index(), input.device().index())
      << "HCCL current stream and tensor must be on the same device: "
      << input.device();
  return stream;
}

void check_out_of_place_buffers(const torch::Tensor& input,
                                const torch::Tensor& output) {
  check_input(output);
  CHECK_EQ(output.device(), input.device()) << "HCCL buffer device mismatch.";
  CHECK_EQ(output.scalar_type(), input.scalar_type())
      << "HCCL buffer dtype mismatch.";
  CHECK(torch::get_overlap_status(input, output) == torch::MemOverlapStatus::No)
      << "Out-of-place AllGather/ReduceScatter buffers must not overlap.";
}

int64_t comm_rank_size(int64_t comm) {
  uint32_t rank_size = 0;
  const HcclResult result =
      HcclGetRankSize(reinterpret_cast<HcclComm>(comm), &rank_size);
  CHECK_EQ(result, HCCL_SUCCESS)
      << "HcclGetRankSize failed, HCCL error " << static_cast<int32_t>(result);
  CHECK_GT(rank_size, 0) << "HCCL communicator has no ranks.";
  return static_cast<int64_t>(rank_size);
}

void check_hccl(HcclResult result, const char* op_name) {
  CHECK_EQ(result, HCCL_SUCCESS)
      << op_name << " failed, HCCL error " << static_cast<int32_t>(result);
}

}  // namespace

void all_reduce_on_current_stream(torch::Tensor& input, int64_t comm) {
  const auto stream = collective_stream(input, comm);
  check_hccl(HcclAllReduce(input.data_ptr(),
                           input.data_ptr(),
                           static_cast<uint64_t>(input.numel()),
                           to_hccl_data_type(input),
                           HCCL_REDUCE_SUM,
                           reinterpret_cast<HcclComm>(comm),
                           stream.stream()),
             "HcclAllReduce");
}

void all_gather_on_current_stream(const torch::Tensor& input,
                                  torch::Tensor& output,
                                  int64_t comm) {
  const auto stream = collective_stream(input, comm);
  check_out_of_place_buffers(input, output);
  const int64_t rank_size = comm_rank_size(comm);
  CHECK_EQ(output.numel() % rank_size, 0)
      << "AllGather output size must be divisible by rank size " << rank_size;
  CHECK_EQ(output.numel() / rank_size, input.numel())
      << "AllGather expects one input-sized output block per rank.";

  check_hccl(HcclAllGather(input.data_ptr(),
                           output.data_ptr(),
                           static_cast<uint64_t>(input.numel()),
                           to_hccl_data_type(input),
                           reinterpret_cast<HcclComm>(comm),
                           stream.stream()),
             "HcclAllGather");
}

void reduce_scatter_on_current_stream(const torch::Tensor& input,
                                      torch::Tensor& output,
                                      int64_t comm) {
  const auto stream = collective_stream(input, comm);
  check_out_of_place_buffers(input, output);
  const int64_t rank_size = comm_rank_size(comm);
  CHECK_EQ(input.numel() % rank_size, 0)
      << "ReduceScatter input size must be divisible by rank size "
      << rank_size;
  CHECK_EQ(input.numel() / rank_size, output.numel())
      << "ReduceScatter expects one output-sized input block per rank.";

  check_hccl(HcclReduceScatter(input.data_ptr(),
                               output.data_ptr(),
                               static_cast<uint64_t>(output.numel()),
                               to_hccl_data_type(input),
                               HCCL_REDUCE_SUM,
                               reinterpret_cast<HcclComm>(comm),
                               stream.stream()),
             "HcclReduceScatter");
}

}  // namespace xllm::kernel::npu
