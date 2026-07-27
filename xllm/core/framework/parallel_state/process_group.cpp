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

#include "core/framework/parallel_state/process_group.h"

#include <algorithm>
#include <functional>
#include <mutex>
#include <unordered_map>

#include "core/platform/device.h"

#if defined(USE_NPU)
#include "core/framework/parallel_state/npu_process_group.h"
#elif defined(USE_MLU)
#include "core/framework/parallel_state/mlu_process_group.h"
#elif defined(USE_CUDA) || defined(USE_DCU)
#include "core/framework/parallel_state/cuda_process_group.h"
#elif defined(USE_MUSA)
#include "core/framework/parallel_state/musa_process_group.h"
#elif defined(USE_ILU)
#include "core/framework/parallel_state/ilu_process_group.h"
#endif

namespace {
std::pair<int, std::vector<uint64_t>> get_trans_group_rank(int world_size,
                                                           int global_rank,
                                                           int split_size) {
  int trans_group_count = split_size;
  int trans_group_size = world_size / split_size;
  int trans_group_index = global_rank % trans_group_size;
  int trans_index = global_rank / trans_group_size;
  std::vector<uint64_t> trans_group_ranks;
  for (int i = 0; i < trans_group_count; i++) {
    uint64_t rank = i * trans_group_size + trans_group_index;
    trans_group_ranks.push_back(rank);
  }

  return {trans_index, trans_group_ranks};
}

std::vector<int64_t> get_gather_shape(int32_t world_size,
                                      const torch::Tensor& input) {
  std::vector<int64_t> out_shape;
  out_shape.reserve(input.dim() + 1);
  out_shape.push_back(world_size);
  for (int64_t dim_size : input.sizes()) {
    out_shape.push_back(dim_size);
  }
  return out_shape;
}

#if defined(USE_NPU)
class PreparedP2POperation final {
 public:
  bool is_recv = false;
  torch::Tensor tensor;
  int64_t peer_rank = -1;
  int32_t tag = 0;
  int64_t payload_bytes = 0;
  bool needs_staging = false;
};

using P2PWave = std::vector<PreparedP2POperation>;
using P2PPostFunction =
    std::function<c10::intrusive_ptr<c10d::Work>(std::vector<torch::Tensor>&,
                                                 int64_t,
                                                 int32_t)>;

class BatchedP2PWork final : public c10d::Work {
 public:
  BatchedP2PWork(std::vector<P2PWave> waves,
                 P2PPostFunction send_function,
                 P2PPostFunction recv_function,
                 std::function<int32_t()> synchronize_function)
      : waves_(std::move(waves)),
        send_function_(std::move(send_function)),
        recv_function_(std::move(recv_function)),
        synchronize_function_(std::move(synchronize_function)) {
    post_next_wave();
  }

  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!completed_) {
      for (const c10::intrusive_ptr<c10d::Work>& work : current_works_) {
        if (work != nullptr && !work->wait(timeout)) {
          return false;
        }
      }
      for (const auto& [destination, staging] : pending_recv_copies_) {
        destination.copy_(staging);
      }
      current_works_.clear();
      retained_tensors_.clear();
      pending_recv_copies_.clear();
      if (next_wave_ == waves_.size()) {
        completed_ = true;
      } else {
        post_next_wave();
      }
    }
    return true;
  }

 private:
  void post_next_wave() {
    CHECK_LT(next_wave_, waves_.size());
    const P2PWave& wave = waves_[next_wave_++];
    std::vector<torch::Tensor> communication_tensors;
    communication_tensors.reserve(wave.size());
    retained_tensors_.reserve(wave.size());
    pending_recv_copies_.reserve(wave.size());
    bool has_send_staging = false;
    for (const PreparedP2POperation& operation : wave) {
      torch::Tensor communication_tensor = operation.tensor;
      if (operation.needs_staging) {
        communication_tensor =
            torch::empty(operation.tensor.sizes(), operation.tensor.options());
        CHECK_EQ(communication_tensor.storage_offset(), 0);
        if (operation.is_recv) {
          pending_recv_copies_.emplace_back(operation.tensor,
                                            communication_tensor);
        } else {
          communication_tensor.copy_(operation.tensor);
          has_send_staging = true;
        }
      }
      communication_tensors.emplace_back(communication_tensor);
      retained_tensors_.emplace_back(std::move(communication_tensor));
    }
    if (has_send_staging) {
      const int32_t synchronize_result = synchronize_function_();
      CHECK_EQ(synchronize_result, 0)
          << "batch_isend_irecv failed to synchronize send staging data.";
    }
    current_works_.reserve(wave.size());
    for (size_t index = 0; index < wave.size(); ++index) {
      const PreparedP2POperation& operation = wave[index];
      std::vector<torch::Tensor> tensor_list = {communication_tensors[index]};
      c10::intrusive_ptr<c10d::Work> work =
          operation.is_recv
              ? recv_function_(tensor_list, operation.peer_rank, operation.tag)
              : send_function_(tensor_list, operation.peer_rank, operation.tag);
      if (work != nullptr) {
        current_works_.emplace_back(std::move(work));
      }
    }
  }

  std::vector<P2PWave> waves_;
  P2PPostFunction send_function_;
  P2PPostFunction recv_function_;
  std::function<int32_t()> synchronize_function_;
  size_t next_wave_ = 0;
  std::vector<c10::intrusive_ptr<c10d::Work>> current_works_;
  std::vector<torch::Tensor> retained_tensors_;
  std::vector<std::pair<torch::Tensor, torch::Tensor>> pending_recv_copies_;
  std::mutex mutex_;
  bool completed_ = false;
};
#endif
}  // namespace

namespace xllm {

ProcessGroup::~ProcessGroup() { shutdown_backend(); }

void ProcessGroup::shutdown_backend() {
  if (pg_ == nullptr) {
    return;
  }
  pg_->shutdown();
  pg_.reset();
}

std::pair<int, std::vector<uint64_t>> get_group_rank(int world_size,
                                                     int global_rank,
                                                     int split_size,
                                                     bool trans) {
  if (trans) {
    return get_trans_group_rank(world_size, global_rank, split_size);
  }
  int target_group_index = global_rank / split_size;
  uint64_t start_rank = target_group_index * split_size;
  uint64_t end_rank = start_rank + split_size;
  std::vector<uint64_t> group_rank;
  int index = global_rank - start_rank;
  for (uint64_t rank = start_rank; rank < end_rank; rank++) {
    group_rank.push_back(rank);
  }
  return {index, group_rank};
}

c10::intrusive_ptr<c10d::Store> create_tcp_store(const std::string& host,
                                                 int port,
                                                 int rank) {
  c10d::TCPStoreOptions tcp_options;
  tcp_options.isServer = (rank == 0);
  tcp_options.port = port;
  return c10::make_intrusive<c10d::TCPStore>(host, tcp_options);
}

void ProcessGroup::allreduce(torch::Tensor& input) {
  CHECK(pg_ != nullptr) << "Process group is not initialized.";
  allreduce_async(input)->wait();
}

c10::intrusive_ptr<c10d::Work> ProcessGroup::allreduce_async(
    torch::Tensor& input) {
  CHECK(pg_ != nullptr) << "Process group is not initialized.";
  std::vector<torch::Tensor> input_tensors = {input};
  return pg_->allreduce(input_tensors);
}

void ProcessGroup::allgather(const torch::Tensor& input,
                             std::vector<torch::Tensor>& outputs) {
  allgather_async(input, outputs)->wait();
}

c10::intrusive_ptr<c10d::Work> ProcessGroup::allgather_async(
    const torch::Tensor& input,
    std::vector<torch::Tensor>& outputs) {
  CHECK(pg_ != nullptr) << "Process group is not initialized.";
  std::vector<torch::Tensor> input_tensors = {input};
  std::vector<std::vector<torch::Tensor>> output_tensors = {outputs};
  return pg_->allgather(output_tensors, input_tensors);
}

c10::intrusive_ptr<c10d::Work> ProcessGroup::allgather_base_async(
    const torch::Tensor& input,
    torch::Tensor& output) {
  CHECK(pg_ != nullptr) << "Process group is not initialized.";
  CHECK_EQ(input.device(), device())
      << "input should be on the same device as the process group";
  CHECK(output.defined()) << "output should be preallocated";
  CHECK_EQ(output.device(), device())
      << "output should be on the same device as the process group";
  CHECK(output.is_contiguous()) << "output should be contiguous";

  torch::Tensor input_buf = input.contiguous();
  const std::vector<int64_t> out_shape =
      get_gather_shape(world_size(), input_buf);
  CHECK_EQ(output.sizes(), torch::IntArrayRef(out_shape))
      << "output shape mismatch for allgather_base_async";
  c10d::AllgatherOptions opts;
  return pg_->_allgather_base(output, input_buf, opts);
}

torch::Tensor ProcessGroup::allgather_base_sync(const torch::Tensor& input) {
  CHECK(pg_ != nullptr) << "Process group is not initialized.";
  CHECK_EQ(input.device(), device())
      << "input should be on the same device as the process group";
  torch::Tensor output =
      torch::empty(get_gather_shape(world_size(), input), input.options());
  allgather_base_async(input, output)->wait();
  return output;
}

void ProcessGroup::reduce_scatter(const torch::Tensor& input,
                                  torch::Tensor& output) {
  CHECK(pg_ != nullptr) << "Process group is not initialized.";
  // make sure input is contiguous
  CHECK(input.is_contiguous()) << "input is not contiguous.";
  std::vector<torch::Tensor> input_tensors = {input};
  std::vector<torch::Tensor> output_tensors = {output};

  c10d::ReduceScatterOptions opts;
  // we use reduce operation SUM for reduce_scatter for default.
  opts.reduceOp = c10d::ReduceOp::SUM;
  pg_->reduce_scatter_tensor_coalesced(output_tensors, input_tensors, opts)
      ->wait();
}

void ProcessGroup::broadcast(torch::Tensor& input, int32_t root_rank) {
  CHECK(pg_ != nullptr) << "Process group is not initialized.";
  // single-rank group: nothing to unify, the local tensor is already the
  // source of truth.
  if (world_size() <= 1) {
    return;
  }
  CHECK(input.is_contiguous()) << "input is not contiguous.";
  std::vector<torch::Tensor> tensors = {input};
  c10d::BroadcastOptions opts;
  opts.rootRank = root_rank;
  pg_->broadcast(tensors, opts)->wait();
}

void ProcessGroup::all_to_all_single(
    torch::Tensor output,
    torch::Tensor input,
    std::vector<int64_t> output_split_sizes,
    std::vector<int64_t> input_split_sizes,
    bool async_op,
    c10::intrusive_ptr<c10d::Work>* async_work) {
  CHECK(pg_ != nullptr) << "Process group is not initialized.";
  CHECK(output.defined())
      << "Output of all_to_all_single function is not defined";
  CHECK(input.defined())
      << "Input of all_to_all_single function is not defined";
  if (input.is_complex()) {
    input = torch::view_as_real(input);
  }
  if (output.is_complex()) {
    output = torch::view_as_real(output);
  }

  auto opts = c10d::AllToAllOptions();
  auto work = pg_->alltoall_base(
      output, input, output_split_sizes, input_split_sizes, opts);
  if (async_op) {
    *async_work = work;
  } else {
    work->wait();
  }
}

void ProcessGroup::send(const torch::Tensor& tensor, int dst, int tag) {
  CHECK(pg_ != nullptr) << "Process group is not initialized.";
  CHECK(tensor.defined()) << "send tensor is not defined";
  std::vector<torch::Tensor> tensors = {tensor};
  pg_->send(tensors, dst, tag)->wait();
}

void ProcessGroup::recv(torch::Tensor& tensor, int src, int tag) {
  CHECK(pg_ != nullptr) << "Process group is not initialized.";
  CHECK(tensor.defined()) << "recv tensor is not defined";
  std::vector<torch::Tensor> tensors = {tensor};
  pg_->recv(tensors, src, tag)->wait();
}

std::string ProcessGroup::hccl_comm_name(bool init_comm) {
  (void)init_comm;
  CHECK(false) << "hccl_comm_name is only supported on NPU HCCL process group.";
  return "";
}

#if defined(USE_NPU)
int64_t ProcessGroup::max_p2p_wave_payload_bytes() const {
  return 256 * 1024 * 1024;
}

int32_t ProcessGroup::synchronize_p2p_staging() {
  return Device(device_).synchronize_default_stream();
}

c10::intrusive_ptr<c10d::Work> ProcessGroup::send_p2p(
    std::vector<torch::Tensor>& tensors,
    int64_t peer_rank,
    int32_t tag) {
  CHECK(pg_ != nullptr) << "P2P send requires an initialized process group.";
  return pg_->send(tensors, peer_rank, tag);
}

c10::intrusive_ptr<c10d::Work> ProcessGroup::recv_p2p(
    std::vector<torch::Tensor>& tensors,
    int64_t peer_rank,
    int32_t tag) {
  CHECK(pg_ != nullptr) << "P2P recv requires an initialized process group.";
  return pg_->recv(tensors, peer_rank, tag);
}

void ProcessGroup::warmup_p2p() {
  std::call_once(p2p_warmup_flag_, [this]() {
    const int32_t group_size = world_size();
    if (group_size <= 1) {
      return;
    }

    const int32_t local_rank = rank();
    torch::Tensor send_tensor = torch::zeros(
        {1}, torch::TensorOptions().dtype(torch::kInt32).device(device()));
    torch::Tensor recv_tensor = torch::empty_like(send_tensor);
    std::vector<std::string> op_types;
    std::vector<torch::Tensor> tensors;
    std::vector<int64_t> remote_ranks;
    const size_t operation_count = static_cast<size_t>(2 * (group_size - 1));
    op_types.reserve(operation_count);
    tensors.reserve(operation_count);
    remote_ranks.reserve(operation_count);
    for (int32_t peer_rank = 0; peer_rank < group_size; ++peer_rank) {
      if (peer_rank == local_rank) {
        continue;
      }
      op_types.emplace_back("send");
      tensors.emplace_back(send_tensor);
      remote_ranks.emplace_back(peer_rank);
      op_types.emplace_back("recv");
      tensors.emplace_back(recv_tensor);
      remote_ranks.emplace_back(peer_rank);
    }
    c10::intrusive_ptr<c10d::Work> work =
        batch_isend_irecv(op_types, tensors, remote_ranks);
    if (work != nullptr) {
      work->wait();
    }
  });
}

c10::intrusive_ptr<c10d::Work> ProcessGroup::batch_isend_irecv(
    std::vector<std::string>& op_types,
    std::vector<torch::Tensor>& tensors,
    std::vector<int64_t> remote_ranks) {
  CHECK_EQ(op_types.size(), tensors.size())
      << "batch_isend_irecv op_types and tensors must align.";
  CHECK_EQ(op_types.size(), remote_ranks.size())
      << "batch_isend_irecv op_types and remote_ranks must align.";

  const int32_t local_rank = rank();
  const int32_t group_size = world_size();
  CHECK_GE(local_rank, 0);
  CHECK_LT(local_rank, group_size);

  for (size_t index = 0; index < op_types.size(); ++index) {
    CHECK(op_types[index] == "recv" || op_types[index] == "send")
        << "batch_isend_irecv op type must be recv or send.";
  }

  constexpr int64_t kMaxChunkBytes = 64 * 1024 * 1024;
  std::vector<PreparedP2POperation> prepared_operations;
  prepared_operations.reserve(op_types.size());
  for (size_t index = 0; index < op_types.size(); ++index) {
    const bool is_recv = op_types[index] == "recv";
    const int64_t peer_rank = remote_ranks[index];
    CHECK_GE(peer_rank, 0);
    CHECK_LT(peer_rank, group_size);
    CHECK_NE(peer_rank, local_rank);
    torch::Tensor tensor = tensors[index];
    CHECK(tensor.defined()) << "batch_isend_irecv tensors must all be defined.";
    const int64_t element_size = static_cast<int64_t>(tensor.element_size());
    CHECK_GT(element_size, 0);
    const int64_t max_chunk_elements = kMaxChunkBytes / element_size;
    CHECK_GT(max_chunk_elements, 0);
    std::function<void(const torch::Tensor&)> append_chunks;
    append_chunks = [&](const torch::Tensor& chunk) {
      if (chunk.numel() <= max_chunk_elements) {
        PreparedP2POperation operation;
        operation.is_recv = is_recv;
        operation.tensor = chunk;
        operation.peer_rank = peer_rank;
        operation.payload_bytes =
            chunk.numel() * static_cast<int64_t>(chunk.element_size());
        operation.needs_staging =
            !chunk.is_contiguous() || chunk.storage_offset() != 0;
        prepared_operations.emplace_back(std::move(operation));
        return;
      }

      int64_t split_dim = -1;
      for (int64_t dim = 0; dim < chunk.dim(); ++dim) {
        if (chunk.size(dim) > 1) {
          split_dim = dim;
          break;
        }
      }
      CHECK_GE(split_dim, 0)
          << "P2P tensor cannot be split below the staging chunk limit.";
      const int64_t split_size = chunk.size(split_dim);
      const int64_t elements_per_slice = chunk.numel() / split_size;
      const int64_t slices_per_chunk =
          std::max<int64_t>(max_chunk_elements / elements_per_slice, 1);
      for (int64_t start = 0; start < split_size; start += slices_per_chunk) {
        const int64_t length = std::min(slices_per_chunk, split_size - start);
        append_chunks(chunk.narrow(split_dim, start, length));
      }
    };
    append_chunks(tensor);
  }

  std::unordered_map<int64_t, int32_t> recv_tags;
  std::unordered_map<int64_t, int32_t> send_tags;
  std::vector<PreparedP2POperation> ordered_operations;
  ordered_operations.reserve(prepared_operations.size());
  auto enqueue = [&](const PreparedP2POperation& source_operation) {
    PreparedP2POperation operation = source_operation;
    const int64_t peer_rank = operation.peer_rank;
    const bool is_recv = operation.is_recv;
    auto& tags = is_recv ? recv_tags : send_tags;
    operation.tag = tags[peer_rank]++;
    ordered_operations.emplace_back(std::move(operation));
  };

  auto enqueue_peer_ops = [&](int32_t peer_rank, bool is_recv) {
    for (const PreparedP2POperation& operation : prepared_operations) {
      if (operation.peer_rank == peer_rank && operation.is_recv == is_recv) {
        enqueue(operation);
      }
    }
  };
  // Independent HCCL P2P calls share one device stream. Posting receives on
  // every rank before sends can block each stream at its first receive, so the
  // matching sends queued behind it never execute. Traverse rank pairs in one
  // global order and give each pair complementary stream order instead.
  for (int32_t low_rank = 0; low_rank < group_size; ++low_rank) {
    for (int32_t high_rank = low_rank + 1; high_rank < group_size;
         ++high_rank) {
      if (local_rank == low_rank) {
        enqueue_peer_ops(high_rank, /*is_recv=*/false);
        enqueue_peer_ops(high_rank, /*is_recv=*/true);
      } else if (local_rank == high_rank) {
        enqueue_peer_ops(low_rank, /*is_recv=*/true);
        enqueue_peer_ops(low_rank, /*is_recv=*/false);
      }
    }
  }
  if (ordered_operations.empty()) {
    return nullptr;
  }

  const int64_t max_wave_payload_bytes = max_p2p_wave_payload_bytes();
  CHECK_GT(max_wave_payload_bytes, 0);
  std::vector<P2PWave> waves;
  P2PWave current_wave;
  int64_t current_wave_payload_bytes = 0;
  for (PreparedP2POperation& operation : ordered_operations) {
    if (!current_wave.empty() &&
        current_wave_payload_bytes + operation.payload_bytes >
            max_wave_payload_bytes) {
      waves.emplace_back(std::move(current_wave));
      current_wave = P2PWave();
      current_wave_payload_bytes = 0;
    }
    current_wave_payload_bytes += operation.payload_bytes;
    current_wave.emplace_back(std::move(operation));
  }
  if (!current_wave.empty()) {
    waves.emplace_back(std::move(current_wave));
  }

  return c10::make_intrusive<BatchedP2PWork>(
      std::move(waves),
      [this](std::vector<torch::Tensor>& wave_tensors,
             int64_t peer_rank,
             int32_t tag) { return send_p2p(wave_tensors, peer_rank, tag); },
      [this](std::vector<torch::Tensor>& wave_tensors,
             int64_t peer_rank,
             int32_t tag) { return recv_p2p(wave_tensors, peer_rank, tag); },
      [this]() { return synchronize_p2p_staging(); });
}
#endif

std::unique_ptr<ProcessGroup> create_process_group(
    int32_t rank,
    int32_t world_size,
    int32_t rank_size,
    int32_t port,
    bool trans,
    const std::string& host,
    const std::string& group_name,
    const torch::Device& device) {
  return std::make_unique<ProcessGroupImpl>(
      rank, world_size, rank_size, port, trans, host, group_name, device);
}

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
// We currently support explicit DiT communication groups on NPU, MLU, and DCU.
// TODO: This function is used by DiT models, since the DiT communication group
// info have already been calculated by rank_generator, we only need to pass the
// info to create the process groups. For any device that want to reuse the
// function and dit process groups, please implement the corresponding
// ProcessGroupImpl construct function.
std::unique_ptr<ProcessGroup> create_process_group(
    int32_t global_rank,
    int32_t local_rank,
    const std::vector<int32_t>& group_ranks,
    int32_t world_size,
    int32_t rank_size,
    int32_t port,
    const std::string& host,
    const std::string& group_name,
    const torch::Device& device) {
  return std::make_unique<ProcessGroupImpl>(global_rank,
                                            local_rank,
                                            group_ranks,
                                            world_size,
                                            rank_size,
                                            port,
                                            host,
                                            group_name,
                                            device);
}
#endif
}  // namespace xllm
