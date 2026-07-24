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

#include "framework/parallel_state/single_process/hccl_process_group.h"

#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUEvent.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <chrono>
#include <utility>
#include <vector>

#include "platform/device.h"

namespace xllm {

namespace {

#define XLLM_HCCLCHECK(cmd)                                            \
  do {                                                                 \
    HcclResult r = (cmd);                                              \
    if (r != HCCL_SUCCESS) {                                           \
      LOG(FATAL) << "HCCL error " << static_cast<int32_t>(r) << " at " \
                 << __FILE__ << ":" << __LINE__;                       \
    }                                                                  \
  } while (0)

// HcclCommInitAll binds each comm to the device context active on the thread
// that created it (the engine thread, with device 0 current). The comms are
// then used from each worker's own thread, where the device must be bound
// eagerly before launching an HCCL op; torch_npu's set_device is lazy and the
// raw HCCL launch is not a torch op that would trigger the deferred bind, so we
// bind explicitly via c10_npu::SetDevice. Without this the runtime reports
// "stream not in current context" and aborts.
void bind_device(const torch::Device& device) {
  CHECK_EQ(c10_npu::SetDevice(static_cast<int32_t>(device.index())),
           ACL_SUCCESS)
      << "c10_npu::SetDevice failed for device " << device.index();
}

HcclDataType to_hccl_data_type(const torch::Tensor& input) {
  switch (input.scalar_type()) {
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
      LOG(FATAL) << "Unsupported tensor dtype for HCCL: "
                 << input.scalar_type();
  }
}

// Lightweight c10d::Work that waits on a single NPU event recorded after the
// HCCL launch. The compute path always queues HCCL on an NPU stream and then
// issues blocking waits via `wait()`, so this is enough to surface completion
// to the caller without taking on the full ProcessGroupHCCL state machine.
// Mirrors CudaNcclWork from the CUDA single-process group.
class HcclNpuWork : public c10d::Work {
 public:
  HcclNpuWork(c10d::OpType op_type,
              c10_npu::NPUStream stream,
              const torch::Device& device)
      : c10d::Work(/*rank=*/-1, op_type), device_(device) {
    event_.record(stream);
  }

  bool isCompleted() override {
    c10_npu::NPUGuard guard(device_);
    return event_.query();
  }

  bool wait(std::chrono::milliseconds /*timeout*/ = kNoTimeout) override {
    c10_npu::NPUGuard guard(device_);
    // Make the current stream wait for completion of the HCCL work that was
    // recorded on the issuing stream. This matches the semantics callers get
    // from c10d_npu::ProcessGroupHCCL where wait() is a stream-side join, not a
    // host-side synchronize.
    event_.block(c10_npu::getCurrentNPUStream(device_.index()));
    return true;
  }

 private:
  c10_npu::NPUEvent event_;
  torch::Device device_;
};

c10::intrusive_ptr<c10d::Work> make_work(c10d::OpType op_type,
                                         c10_npu::NPUStream stream,
                                         const torch::Device& device) {
  return c10::make_intrusive<HcclNpuWork>(op_type, stream, device);
}

}  // namespace

HcclProcessGroup::HcclProcessGroup(int32_t rank,
                                   int32_t world_size,
                                   const torch::Device& device,
                                   HcclComm comm)
    : ProcessGroup(rank, world_size, device), comm_(comm) {
  CHECK(comm != nullptr) << "HcclComm must be non-null";
}

HcclProcessGroup::~HcclProcessGroup() {
  // We own the comm; release it before the base class destructor runs so any
  // device state can be torn down cleanly.
  bind_device(device());
  HcclCommDestroy(comm_);
  comm_ = nullptr;
  Device::empty_cache(device().index());
}

c10_npu::NPUStream HcclProcessGroup::hccl_stream() {
  return c10_npu::getCurrentNPUStream(device().index());
}

void HcclProcessGroup::allreduce(torch::Tensor& input) {
  allreduce_async(input)->wait();
}

c10::intrusive_ptr<c10d::Work> HcclProcessGroup::allreduce_async(
    torch::Tensor& input) {
  CHECK_EQ(input.device(), device())
      << "allreduce input must live on the process group's device";
  CHECK(input.is_contiguous()) << "allreduce input must be contiguous";

  bind_device(device());
  c10_npu::NPUStream stream = hccl_stream();
  XLLM_HCCLCHECK(HcclAllReduce(input.data_ptr(),
                               input.data_ptr(),
                               static_cast<uint64_t>(input.numel()),
                               to_hccl_data_type(input),
                               HCCL_REDUCE_SUM,
                               comm_,
                               stream.stream()));
  return make_work(c10d::OpType::ALLREDUCE, stream, device());
}

void HcclProcessGroup::allgather(const torch::Tensor& input,
                                 std::vector<torch::Tensor>& outputs) {
  allgather_async(input, outputs)->wait();
}

c10::intrusive_ptr<c10d::Work> HcclProcessGroup::allgather_async(
    const torch::Tensor& input,
    std::vector<torch::Tensor>& outputs) {
  CHECK_EQ(static_cast<int32_t>(outputs.size()), world_size())
      << "allgather output count must equal world_size";
  CHECK_EQ(input.device(), device())
      << "allgather input must live on the process group's device";
  CHECK(input.is_contiguous()) << "allgather input must be contiguous";

  bind_device(device());
  c10_npu::NPUStream stream = hccl_stream();
  // HcclAllGather requires a single contiguous receive buffer; build one
  // sized [world_size, *input.shape] then copy into the per-rank outputs
  // afterwards. This mirrors the ProcessGroup::allgather + cat semantics
  // callers expect.
  std::vector<int64_t> stacked_shape;
  stacked_shape.reserve(input.dim() + 1);
  stacked_shape.push_back(world_size());
  for (int64_t s : input.sizes()) {
    stacked_shape.push_back(s);
  }
  torch::Tensor stacked = torch::empty(stacked_shape, input.options());
  XLLM_HCCLCHECK(HcclAllGather(input.data_ptr(),
                               stacked.data_ptr(),
                               static_cast<uint64_t>(input.numel()),
                               to_hccl_data_type(input),
                               comm_,
                               stream.stream()));
  // Slice stacked into the supplied outputs. Each slice shares storage with
  // stacked, so the gather kernel only writes once.
  for (int32_t i = 0; i < world_size(); ++i) {
    if (!outputs[i].defined()) {
      outputs[i] = stacked.select(0, i);
    } else {
      outputs[i].copy_(stacked.select(0, i), /*non_blocking=*/true);
    }
  }
  return make_work(c10d::OpType::ALLGATHER, stream, device());
}

c10::intrusive_ptr<c10d::Work> HcclProcessGroup::allgather_base_async(
    const torch::Tensor& input,
    torch::Tensor& output) {
  CHECK_EQ(input.device(), device())
      << "allgather_base input must live on the process group's device";
  CHECK(output.defined()) << "allgather_base output must be preallocated";
  CHECK_EQ(output.device(), device())
      << "allgather_base output must live on the process group's device";
  CHECK(output.is_contiguous()) << "allgather_base output must be contiguous";

  torch::Tensor input_buf = input.contiguous();
  CHECK_EQ(output.numel(), input_buf.numel() * world_size())
      << "allgather_base output size must equal world_size * input size";

  bind_device(device());
  c10_npu::NPUStream stream = hccl_stream();
  XLLM_HCCLCHECK(HcclAllGather(input_buf.data_ptr(),
                               output.data_ptr(),
                               static_cast<uint64_t>(input_buf.numel()),
                               to_hccl_data_type(input_buf),
                               comm_,
                               stream.stream()));
  return make_work(c10d::OpType::_ALLGATHER_BASE, stream, device());
}

torch::Tensor HcclProcessGroup::allgather_base_sync(
    const torch::Tensor& input) {
  CHECK_EQ(input.device(), device())
      << "allgather_base input must live on the process group's device";
  std::vector<int64_t> out_shape;
  out_shape.reserve(input.dim() + 1);
  out_shape.push_back(world_size());
  for (int64_t s : input.sizes()) {
    out_shape.push_back(s);
  }
  torch::Tensor output = torch::empty(out_shape, input.options());
  allgather_base_async(input, output)->wait();
  return output;
}

void HcclProcessGroup::reduce_scatter(const torch::Tensor& input,
                                      torch::Tensor& output) {
  CHECK(input.is_contiguous()) << "reduce_scatter input must be contiguous";
  CHECK_EQ(input.device(), device())
      << "reduce_scatter input must live on the process group's device";
  CHECK(output.defined()) << "reduce_scatter output must be defined";
  CHECK_EQ(output.device(), device())
      << "reduce_scatter output must live on the process group's device";
  CHECK_EQ(input.numel(), output.numel() * world_size())
      << "reduce_scatter input size must equal world_size * output size";

  bind_device(device());
  c10_npu::NPUStream stream = hccl_stream();
  XLLM_HCCLCHECK(HcclReduceScatter(input.data_ptr(),
                                   output.data_ptr(),
                                   static_cast<uint64_t>(output.numel()),
                                   to_hccl_data_type(input),
                                   HCCL_REDUCE_SUM,
                                   comm_,
                                   stream.stream()));
  // Block the caller until the scatter completes so the semantics match the
  // base class (which calls ->wait()).
  make_work(c10d::OpType::REDUCE_SCATTER, stream, device())->wait();
}

void HcclProcessGroup::all_to_all_single(
    torch::Tensor output,
    torch::Tensor input,
    std::vector<int64_t> output_split_sizes,
    std::vector<int64_t> input_split_sizes,
    bool async_op,
    c10::intrusive_ptr<c10d::Work>* async_work) {
  CHECK(output.defined()) << "all_to_all_single output must be defined";
  CHECK(input.defined()) << "all_to_all_single input must be defined";
  CHECK_EQ(input.device(), device())
      << "all_to_all_single input must live on the process group's device";
  CHECK_EQ(output.device(), device())
      << "all_to_all_single output must live on the process group's device";

  // Treat complex tensors the same way the base class does: split each
  // complex element into real+imag along the last dim before sending.
  if (input.is_complex()) {
    input = torch::view_as_real(input);
  }
  if (output.is_complex()) {
    output = torch::view_as_real(output);
  }
  CHECK(input.is_contiguous()) << "all_to_all_single input must be contiguous";
  CHECK(output.is_contiguous())
      << "all_to_all_single output must be contiguous";

  const int32_t ws = world_size();
  std::vector<int64_t> in_splits = input_split_sizes;
  std::vector<int64_t> out_splits = output_split_sizes;
  if (in_splits.empty()) {
    CHECK_EQ(input.size(0) % ws, 0)
        << "input dim 0 must be divisible by world_size for equal-split a2a";
    in_splits.assign(ws, input.size(0) / ws);
  }
  if (out_splits.empty()) {
    CHECK_EQ(output.size(0) % ws, 0)
        << "output dim 0 must be divisible by world_size for equal-split a2a";
    out_splits.assign(ws, output.size(0) / ws);
  }
  CHECK_EQ(static_cast<int32_t>(in_splits.size()), ws);
  CHECK_EQ(static_cast<int32_t>(out_splits.size()), ws);

  // HCCL has no direct alltoall-with-splits primitive; emulate via a batch of
  // send/recv items submitted together through HcclBatchSendRecv (the NPU
  // analogue of ncclGroupStart/ncclSend/ncclRecv/ncclGroupEnd).
  const int64_t in_inner = input.numel() / std::max<int64_t>(1, input.size(0));
  const int64_t out_inner =
      output.numel() / std::max<int64_t>(1, output.size(0));
  const HcclDataType dtype = to_hccl_data_type(input);
  const size_t elem_size = static_cast<size_t>(input.element_size());
  char* input_ptr = static_cast<char*>(input.data_ptr());
  char* output_ptr = static_cast<char*>(output.data_ptr());

  std::vector<HcclSendRecvItem> items;
  items.reserve(static_cast<size_t>(ws) * 2);
  int64_t in_offset = 0;
  int64_t out_offset = 0;
  for (int32_t r = 0; r < ws; ++r) {
    const uint64_t send_count = static_cast<uint64_t>(in_splits[r] * in_inner);
    const uint64_t recv_count =
        static_cast<uint64_t>(out_splits[r] * out_inner);
    if (send_count > 0) {
      items.push_back(HcclSendRecvItem{HCCL_SEND,
                                       input_ptr + in_offset * elem_size,
                                       send_count,
                                       dtype,
                                       static_cast<uint32_t>(r)});
    }
    if (recv_count > 0) {
      items.push_back(HcclSendRecvItem{HCCL_RECV,
                                       output_ptr + out_offset * elem_size,
                                       recv_count,
                                       dtype,
                                       static_cast<uint32_t>(r)});
    }
    in_offset += in_splits[r] * in_inner;
    out_offset += out_splits[r] * out_inner;
  }

  bind_device(device());
  c10_npu::NPUStream stream = hccl_stream();
  XLLM_HCCLCHECK(HcclBatchSendRecv(items.data(),
                                   static_cast<uint32_t>(items.size()),
                                   comm_,
                                   stream.stream()));

  c10::intrusive_ptr<c10d::Work> work =
      make_work(c10d::OpType::ALLTOALL_BASE, stream, device());
  if (async_op) {
    CHECK(async_work != nullptr) << "async_work must be provided for async_op";
    *async_work = work;
  } else {
    work->wait();
  }
}

}  // namespace xllm
