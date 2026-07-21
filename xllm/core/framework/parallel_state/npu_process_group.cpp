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
#include "npu_process_group.h"

#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>

#include <c10d/ProcessGroup.hpp>
#include <c10d/TCPStore.hpp>
#include <torch_npu/csrc/distributed/ProcessGroupHCCL.hpp>

#include "core/framework/config/dit_config.h"
#include "core/framework/config/eplb_config.h"
#include "core/util/env_var.h"
#include "npu_rank_table_env.h"
#include "platform/device.h"

namespace {
inline bool is_npu(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return false;
  }
  return tensor.device().is_privateuseone();
}

torch::Tensor flatten_for_scatter_gather(std::vector<torch::Tensor>& tensors) {
  auto& t = tensors[0];
  std::vector<int64_t> sizes{static_cast<int64_t>(tensors.size())};
  sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
  return torch::empty(sizes, t.options());
}

HcclDataType to_hccl_data_type(const torch::Tensor& input) {
  const auto type = input.scalar_type();
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
  }
}

void check_input(torch::Tensor input) {
  CHECK(is_npu(input)) << "input should be npu tensor";
  CHECK(input.is_contiguous()) << "input should be contiguous";
  CHECK(!input.is_sparse()) << "input have to be npu dense tensor";
}

std::string resolve_tcp_store_host(const std::string& host, int32_t rank_size) {
  // A rank_size=1 group is local to the current worker process. Using the
  // cluster master address here makes remote workers connect back to rank0's
  // node for their private group and can deadlock startup.
  return rank_size == 1 ? "127.0.0.1" : host;
}

constexpr uint32_t kHcclAivExpansionMode = 3;

void configure_hccl_aiv_expansion(
    const std::string& group_name,
    int32_t rank_size,
    const c10::intrusive_ptr<c10d_npu::ProcessGroupHCCL::Options>& options) {
  if (group_name != "tp_group" || rank_size <= 1) {
    return;
  }
  const auto aiv_mode = xllm::util::get_optional_string_env("XLLM_HCCL_TP_AIV");
  if (!aiv_mode ||
      (*aiv_mode != "1" && *aiv_mode != "true" && *aiv_mode != "aiv")) {
    return;
  }

  // CANN restricts AIV expansion across multiple communicators, so scope it
  // to the multi-rank tensor-parallel communicator used by this optimization.
  options->hccl_config["hccl_op_expansion_mode"] = kHcclAivExpansionMode;
  LOG(INFO) << "Enabling HCCL AIV expansion for " << group_name << ".";
}
}  // namespace

namespace xllm {

ProcessGroupImpl::ProcessGroupImpl(int32_t global_rank,
                                   int32_t world_size,
                                   int32_t rank_size,
                                   int32_t port,
                                   bool trans,
                                   const std::string& host,
                                   const std::string& group_name,
                                   const torch::Device& device)
    : ProcessGroup(global_rank, world_size, device),
      comm_stream_(c10_npu::getNPUStreamFromPool(device.index())) {
  parallel_state::sync_torch_npu_rank_table_file_env(
      ::xllm::EPLBConfig::get_instance().rank_tablefile());
  c10::intrusive_ptr<c10d_npu::ProcessGroupHCCL::Options> hccl_pg_options =
      c10d_npu::ProcessGroupHCCL::Options::create();
  hccl_pg_options->group_id = group_name;

  int32_t rank = global_rank;
  if (world_size != rank_size) {
    auto [local_rank, group_ranks] =
        get_group_rank(world_size, global_rank, rank_size, trans);
    std::vector<uint32_t> uint32_ranks;
    for (auto rank : group_ranks) {
      uint32_ranks.push_back(static_cast<uint32_t>(rank));
    }
    hccl_pg_options->global_ranks_in_group = uint32_ranks;
    rank = local_rank;
  }
  // Single-rank process groups do not need rendezvous with another worker.
  // Use an ephemeral localhost port to avoid collisions with stale TCPStore
  // listeners from previous abnormal exits in dense same-host launches.
  const int32_t store_port = rank_size == 1 ? 0 : port;
  auto store = create_tcp_store(
      resolve_tcp_store_host(host, rank_size), store_port, rank);
  configure_hccl_aiv_expansion(group_name, rank_size, hccl_pg_options);
  pg_ = std::make_unique<c10d_npu::ProcessGroupHCCL>(
      store, rank, rank_size, hccl_pg_options);
}

ProcessGroupImpl::ProcessGroupImpl(int32_t global_rank,
                                   int32_t local_rank,
                                   const std::vector<int32_t>& group_ranks,
                                   int32_t world_size,
                                   int32_t rank_size,
                                   int32_t port,
                                   const std::string& host,
                                   const std::string& group_name,
                                   const torch::Device& device)
    : ProcessGroup(global_rank, world_size, device),
      comm_stream_(c10_npu::getNPUStreamFromPool(device.index())) {
  parallel_state::sync_torch_npu_rank_table_file_env(
      ::xllm::EPLBConfig::get_instance().rank_tablefile());
  c10::intrusive_ptr<c10d_npu::ProcessGroupHCCL::Options> hccl_pg_options =
      c10d_npu::ProcessGroupHCCL::Options::create();
  hccl_pg_options->group_id = group_name;
  if (world_size != rank_size) {
    std::vector<uint32_t> uint32_ranks;
    for (auto rank : group_ranks) {
      uint32_ranks.push_back(static_cast<uint32_t>(rank));
    }
    hccl_pg_options->global_ranks_in_group = uint32_ranks;
  }

  if (::xllm::DiTConfig::get_instance().dit_debug_print()) {
    std::stringstream ranks_ss;
    ranks_ss << "Group : [" << group_ranks[0];
    for (size_t i = 1; i < group_ranks.size(); i++) {
      ranks_ss << ", " << group_ranks[i];
    }
    ranks_ss << "]" << std::endl;

    LOG(INFO) << "Creating HccLProcessGroup for " << group_name
              << " group, with global rank " << global_rank << ", local rank"
              << local_rank << ", with port " << host << ":" << port
              << ", rank_size is " << rank_size << ", world_size is "
              << world_size
              << ", the following ranks should share the same port, "
              << ranks_ss.str();
  }

  const int32_t store_port = rank_size == 1 ? 0 : port;
  auto store = create_tcp_store(
      resolve_tcp_store_host(host, rank_size), store_port, local_rank);
  configure_hccl_aiv_expansion(group_name, rank_size, hccl_pg_options);
  pg_ = std::make_unique<c10d_npu::ProcessGroupHCCL>(
      store, local_rank, rank_size, hccl_pg_options);
}

// Destructor.
ProcessGroupImpl::~ProcessGroupImpl() {
  if (pg_) {
    shutdown_backend();
  } else if (comm_ != nullptr) {
    HCCLCHECK(HcclCommDestroy(comm_));
    comm_ = nullptr;
  }
  Device::empty_cache(device().index());
}

ProcessGroupImpl::ProcessGroupImpl(int rank,
                                   int world_size,
                                   const torch::Device& device,
                                   HcclComm comm)
    : ProcessGroup(rank, world_size, device),
      comm_(comm),
      comm_stream_(c10_npu::getNPUStreamFromPool(device.index())) {}

std::string ProcessGroupImpl::hccl_comm_name(bool init_comm) {
  CHECK(pg_ != nullptr) << "HCCL comm name requires a torch NPU process group.";
#if defined(USE_NPU) &&         \
    (TORCH_VERSION_MAJOR < 2 || \
     (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 7))
  return pg_->getHcclCommName(pg_->getRank(), init_comm);
#else
  auto* hccl_pg = dynamic_cast<c10d_npu::ProcessGroupHCCL*>(pg_.get());
  CHECK(hccl_pg != nullptr) << "Process group is not NPU HCCL.";
  return hccl_pg->getHcclCommName(pg_->getRank(), init_comm);
#endif
}

}  // namespace xllm
