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

#include "core/framework/parallel_state/collective_communicator.h"

#include <algorithm>

#include "core/framework/parallel_state/mapping_npu.h"

#if defined(USE_NPU)
#include "core/framework/parallel_state/npu_process_group.h"
#include "xllm_atb_layers/core/include/atb_speed/base/external_comm_manager.h"
#include "xllm_atb_layers/core/include/atb_speed/utils/singleton.h"
#elif defined(USE_MLU)
#include "mlu_process_group.h"
#elif defined(USE_CUDA) || defined(USE_DCU)
#include "cuda_process_group.h"
#elif defined(USE_ILU)
#include "ilu_process_group.h"
#elif defined(USE_MUSA)
#include "musa_process_group.h"
#endif
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/parallel_config.h"
#include "parallel_args.h"
#include "parallel_state.h"
#include "process_group.h"
#include "util/json_reader.h"
#include "util/net.h"

namespace xllm {

#if defined(USE_NPU)
namespace {

bool rank_table_rank_id_matches(const nlohmann::json& rank_id,
                                const std::string& target_rank_id) {
  if (rank_id.is_string()) {
    return rank_id.get<std::string>() == target_rank_id;
  }
  if (rank_id.is_number_integer()) {
    return std::to_string(rank_id.get<int64_t>()) == target_rank_id;
  }
  return false;
}

std::string get_rank_table_server_host(int32_t global_rank,
                                       const std::string& fallback_host) {
  const std::string& rank_tablefile =
      ::xllm::EPLBConfig::get_instance().rank_tablefile();
  if (rank_tablefile.empty()) {
    return fallback_host;
  }

  JsonReader rank_table_reader;
  if (!rank_table_reader.parse(rank_tablefile)) {
    return fallback_host;
  }

  const nlohmann::json rank_table = rank_table_reader.data();
  if (!rank_table.is_object()) {
    return fallback_host;
  }

  const std::string target_rank_id = std::to_string(global_rank);
  auto search_server_list =
      [&target_rank_id](const nlohmann::json& server_list) {
        if (!server_list.is_array()) {
          return std::string();
        }

        for (const nlohmann::json& server : server_list) {
          if (!server.is_object()) {
            continue;
          }

          auto server_id = server.find("server_id");
          auto devices = server.find("device");
          if (server_id == server.end() || !server_id->is_string() ||
              server_id->get<std::string>().empty() ||
              devices == server.end() || !devices->is_array()) {
            continue;
          }

          for (const nlohmann::json& device : *devices) {
            if (!device.is_object()) {
              continue;
            }

            auto rank_id = device.find("rank_id");
            if (rank_id != device.end() &&
                rank_table_rank_id_matches(*rank_id, target_rank_id)) {
              return server_id->get<std::string>();
            }
          }
        }
        return std::string();
      };

  auto server_list = rank_table.find("server_list");
  if (server_list != rank_table.end()) {
    const std::string host = search_server_list(*server_list);
    if (!host.empty()) {
      return host;
    }
  }

  auto group_list = rank_table.find("group_list");
  if (group_list != rank_table.end() && group_list->is_array()) {
    for (const nlohmann::json& group : *group_list) {
      if (!group.is_object()) {
        continue;
      }

      server_list = group.find("server_list");
      if (server_list == group.end()) {
        continue;
      }

      const std::string host = search_server_list(*server_list);
      if (!host.empty()) {
        return host;
      }
    }
  }

  return fallback_host;
}

struct DispatchAndCombineComm {
  nlohmann::json mapping_data;
  atb_speed::base::Mapping mapping;
  std::string domain;
  HcclComm comm = nullptr;
};

DispatchAndCombineComm create_dispatch_and_combine_comm(int32_t global_rank,
                                                        int32_t world_size,
                                                        int32_t dp_size,
                                                        int32_t ep_size,
                                                        int32_t cp_size) {
  const int32_t normalized_cp_size = cp_size > 0 ? cp_size : 1;
  const int32_t attn_tp_size = world_size / (dp_size * normalized_cp_size);

  MappingNPU::Options mapping_options;
  mapping_options.dp_size(dp_size)
      .tp_size(attn_tp_size)
      .moe_tp_size(world_size / ep_size)
      .moe_ep_size(ep_size)
      .pp_size(1)
      .sp_size(1)
      .cp_size(normalized_cp_size);

  MappingNPU mapping_npu(EPLBConfig::get_instance().rank_tablefile(),
                         world_size,
                         global_rank,
                         mapping_options);
  DispatchAndCombineComm result;
  result.mapping_data = mapping_npu.to_json();
  result.mapping.ParseParam(result.mapping_data);
  result.mapping.InitGlobalCommDomain(
      ParallelConfig::get_instance().communication_backend());

  auto moe_ep_parallel_info = result.mapping.Get(atb_speed::base::MOE_EP);
  const bool moe_ep_is_world =
      moe_ep_parallel_info.rankIds.size() == static_cast<size_t>(world_size);
  const uint32_t comm_buffer_size =
      moe_ep_is_world ? 0 : moe_ep_parallel_info.bufferSize;
  const bool reuse_comm_domain = moe_ep_is_world;
  result.domain =
      atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
          moe_ep_parallel_info.groupId,
          moe_ep_parallel_info.rankIds,
          moe_ep_parallel_info.rank,
          ParallelConfig::get_instance().communication_backend(),
          comm_buffer_size,
          0,
          reuse_comm_domain);
  result.comm =
      atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommPtr(
          result.domain);
  return result;
}

}  // namespace
#endif

CollectiveCommunicator::CollectiveCommunicator(int global_rank,
                                               int world_size,
                                               int dp_size,
                                               int ep_size,
                                               int cp_size)
    : CollectiveCommunicatorBase(global_rank, world_size) {
#if defined(USE_NPU)
  // create hccl process group with hccl_root_info
  // std::vector<HcclRootInfo> unique_ids;
  // for (const auto& protoId : uids.comm_unique_ids()) {
  //   HcclRootInfo id;
  //   std::memcpy(
  //       id.internal, protoId.comm_unique_id().data(), sizeof(id.internal));
  //   unique_ids.push_back(id);
  // }
  // HcclComm comm;
  // auto hccl_result = HcclCommInitRootInfo(
  //     world_size, &unique_ids[0], global_rank, &comm);
  // CHECK(hccl_result == HCCL_SUCCESS)
  //     << "HcclCommInitRootInfo failed, global rank is " <<
  //     global_rank;
  // std::unique_ptr<ProcessGroupHCCL> hccl_pg =
  //     std::make_unique<ProcessGroupHCCL>(
  //         global_rank, world_size, device, comm);

  // comunicator will be inited in torch.
  if (::xllm::KernelConfig::get_instance().npu_kernel_backend() == "TORCH") {
    parallel_args_ = std::make_unique<ParallelArgs>(
        global_rank, world_size, dp_size, cp_size, nullptr, ep_size);
    parallel_args_->kv_split_size(
        ::xllm::ParallelConfig::get_instance().kv_split_size());
    return;
  }

  // comunicator will be inited in atb.
  // HACK: MappingNPU internally uses a static counter to auto-assign
  // buffer_offset for multi-model scenarios. This is a hack and should be
  // refactored later.
  const int32_t normalized_cp_size = cp_size > 0 ? cp_size : 1;
  const int32_t attn_tp_size = world_size / (dp_size * normalized_cp_size);
  // FLAGS_kv_split_size: 0 -> leave Options::kv_split_size = -1 so that
  // MappingNPU falls back to cp_size (byte-equivalent). >0 -> propagate
  // verbatim; MappingNPU::validate() enforces divisibility against cp_size.
  const int32_t kv_split_size =
      ::xllm::ParallelConfig::get_instance().kv_split_size();
  const int32_t mapping_kv_split_size = kv_split_size > 0 ? kv_split_size : -1;
  MappingNPU::Options mapping_options;
  mapping_options.dp_size(dp_size)
      .tp_size(attn_tp_size)
      .moe_tp_size(world_size / ep_size)
      .moe_ep_size(ep_size)
      .pp_size(1)
      .sp_size(1)
      .cp_size(normalized_cp_size)
      .kv_split_size(mapping_kv_split_size);
  MappingNPU mapping_npu(::xllm::EPLBConfig::get_instance().rank_tablefile(),
                         world_size,
                         global_rank,
                         mapping_options);
  auto mapping_data = mapping_npu.to_json();
  atb_speed::base::Mapping mapping;
  mapping.ParseParam(mapping_data);
  mapping.InitGlobalCommDomain(
      ::xllm::ParallelConfig::get_instance().communication_backend());
  auto moeEpParallelInfo = mapping.Get(atb_speed::base::MOE_EP);
  auto dispatchAndCombinecommDomain =
      atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
          moeEpParallelInfo.groupId,
          moeEpParallelInfo.rankIds,
          moeEpParallelInfo.rank,
          ::xllm::ParallelConfig::get_instance().communication_backend(),
          moeEpParallelInfo.bufferSize,
          false);
  auto dispatchAndCombineHcclComm =
      atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommPtr(
          dispatchAndCombinecommDomain);
  parallel_args_ = std::make_unique<ParallelArgs>(global_rank,
                                                  world_size,
                                                  dp_size,
                                                  nullptr,
                                                  ep_size,
                                                  cp_size,
                                                  mapping_data,
                                                  mapping,
                                                  dispatchAndCombinecommDomain,
                                                  dispatchAndCombineHcclComm);
  parallel_args_->kv_split_size(
      ::xllm::ParallelConfig::get_instance().kv_split_size());
#else
  parallel_args_ = std::make_unique<ParallelArgs>(
      global_rank, world_size, dp_size, cp_size, nullptr, ep_size);
  parallel_args_->kv_split_size(
      ::xllm::ParallelConfig::get_instance().kv_split_size());
#endif
}

void CollectiveCommunicator::create_process_groups(
    const std::string& master_addr,
    const torch::Device& device) {
  int32_t global_rank = parallel_args_->rank();
  int32_t world_size = parallel_args_->world_size();
  int32_t dp_size = parallel_args_->dp_size();
  int32_t ep_size = parallel_args_->ep_size();
  int32_t cp_size = parallel_args_->cp_size();

  std::string host;
  int32_t port;
  net::parse_host_port_from_addr(master_addr, host, port);

  int32_t port_offset = 0;

  // Encoder DP is used by multi-modal models to parallelize vision encoder
  // work inside each language-model TP group. The rank set matches the TP
  // group, but each rank runs a full encoder on different multi-modal items.
  if (::xllm::ParallelConfig::get_instance().enable_mm_encoder_dp()) {
    const int32_t encoder_dp_size = world_size / dp_size;
    port_offset = global_rank / encoder_dp_size + 1;
    encoder_dp_group_ = create_process_group(global_rank,
                                             world_size,
                                             encoder_dp_size,
                                             port + port_offset,
                                             false,
                                             host,
                                             "encoder_dp_group",
                                             device);
    parallel_args_->encoder_dp_group_ = encoder_dp_group_.get();
    port += dp_size;
  }

#if defined(USE_NPU)
  if (::xllm::KernelConfig::get_instance().npu_kernel_backend() == "ATB") {
    // ATB owns TP/DP/EP; build a standalone HCCL CP ProcessGroup for
    // model-side AllGather.
    if (cp_size > 1) {
      const std::vector<int32_t> cp_ranks =
          parallel_state::compute_cp_group_ranks(
              global_rank, world_size, dp_size, cp_size);
      const int32_t cp_local_rank = parallel_args_->cp_rank();
      CHECK_EQ(cp_ranks.size(), cp_size);
      CHECK_GE(cp_local_rank, 0);
      CHECK_LT(cp_local_rank, cp_size);
      CHECK_EQ(cp_ranks[cp_local_rank], global_rank);
      // Unique TCPStore port per CP group (keyed by attn TP rank).
      const int32_t attn_tp_size = world_size / (dp_size * cp_size);
      const int32_t tp_rank = global_rank % attn_tp_size;
      cp_group_ = create_process_group(global_rank,
                                       cp_local_rank,
                                       cp_ranks,
                                       world_size,
                                       cp_size,
                                       port + 1 + tp_rank,
                                       host,
                                       "cp_group",
                                       device);
      parallel_args_->cp_group_ = cp_group_.get();
    }
    return;
  }
#endif

  const int32_t world_group_port = ++port;
  process_group_ = create_process_group(global_rank,
                                        world_size,
                                        world_size,
                                        world_group_port,
                                        false,
                                        host,
                                        "world_group",
                                        device);
  parallel_args_->process_group_ = process_group_.get();
  parallel_args_->python_rendezvous_host_ = host;
  parallel_args_->python_rendezvous_port_ = world_group_port;

  // Orthogonal CP x TP (NPU TORCH only): the rank layout is
  //   rank = dp_rank * (cp_size * tp_size) + cp_rank * tp_size + tp_rank
  // so tensor parallelism spans world_size / (dp_size * cp_size), NOT
  // world_size / dp_size. Narrowing tp_size here is what makes attention head
  // sharding and the o_proj all-reduce use the attention-TP width: consumers
  // read it back through tp_group_->world_size(). Leaving it at world/dp would
  // make ranks r and r + tp_size hold the same heads and double-accumulate in
  // the all-reduce, which is a silent numerical error rather than a crash.
  const int32_t normalized_cp_size = cp_size > 0 ? cp_size : 1;
  bool use_orthogonal_cp = false;
#if defined(USE_NPU)
  use_orthogonal_cp =
      ::xllm::KernelConfig::get_instance().npu_kernel_backend() == "TORCH" &&
      normalized_cp_size > 1;
#endif
  int32_t tp_size = use_orthogonal_cp
                        ? world_size / (dp_size * normalized_cp_size)
                        : world_size / dp_size;
  CHECK_GT(tp_size, 0) << "attention tp_size must be positive: world_size="
                       << world_size << ", dp_size=" << dp_size
                       << ", cp_size=" << normalized_cp_size;
  CHECK_EQ(tp_size * dp_size * (use_orthogonal_cp ? normalized_cp_size : 1),
           world_size)
      << "world_size (" << world_size << ") must equal dp_size * cp_size * "
      << "tp_size (" << dp_size << " * " << normalized_cp_size << " * "
      << tp_size << ")";
  // Group counts stop tracking dp_size once tp_size narrows, so derive every
  // TCPStore window from the group width instead of assuming world/dp.
  const int32_t tp_group_count = world_size / tp_size;
  port_offset = global_rank / tp_size + 1;
  std::string tp_host = host;
#if defined(USE_NPU)
  if (::xllm::KernelConfig::get_instance().npu_kernel_backend() == "TORCH" &&
      tp_group_count > 1) {
    const int32_t tp_group_start = (global_rank / tp_size) * tp_size;
    tp_host = get_rank_table_server_host(tp_group_start, host);
  }
#endif
  tp_group_ = create_process_group(global_rank,
                                   world_size,
                                   tp_size,
                                   port + port_offset,
                                   false,
                                   tp_host,
                                   "tp_group",
                                   device);
  parallel_args_->tp_group_ = tp_group_.get();
  // Publish the narrowed width so consumers that read tp_size() (rather than
  // tp_group_->world_size()) agree with the group actually created above.
  parallel_args_->tp_size(tp_size);
  // Single-rank group is used for modules that don't need tensor parallel (TP)
  // communication. This avoids unnecessary communication. When tp_size > 1,
  // create a process group of size 1 for each rank. Otherwise, reuse tp_group
  // for single-rank operations.
  int32_t single_rank_group_count = 0;
  int32_t single_rank_group_port_gap = 0;
  if (tp_size > 1) {
    // Keep local single-rank TCPStore ports away from the multi-rank group
    // window. Otherwise the last single-rank port can sit directly on the next
    // group's base port and hit EADDRINUSE in dense same-host launches.
    single_rank_group_port_gap = world_size;
    single_rank_group_ = create_process_group(
        global_rank,
        world_size,
        1,
        port + tp_group_count + single_rank_group_port_gap + global_rank + 1,
        false,
        host,
        "single_rank_group",
        device);
    parallel_args_->single_rank_group_ = single_rank_group_.get();
    single_rank_group_count = world_size;
  } else {
    parallel_args_->single_rank_group_ = tp_group_.get();
  }
  port += tp_group_count + single_rank_group_port_gap + single_rank_group_count;

#if defined(USE_NPU)
  if (use_orthogonal_cp) {
    // A CP group varies cp_rank while holding (dp_rank, tp_rank) fixed, so its
    // members are strided by tp_size and cannot be expressed by the contiguous
    // or `trans` groupings of the size-only overload. Enumerate the ranks
    // explicitly instead.
    const std::vector<int32_t> cp_ranks =
        parallel_state::compute_cp_group_ranks(
            global_rank, world_size, dp_size, normalized_cp_size);
    const int32_t cp_local_rank = parallel_args_->cp_rank();
    CHECK_EQ(static_cast<int32_t>(cp_ranks.size()), normalized_cp_size);
    CHECK_GE(cp_local_rank, 0);
    CHECK_LT(cp_local_rank, normalized_cp_size);
    CHECK_EQ(cp_ranks[cp_local_rank], global_rank)
        << "cp_rank() must index this rank inside its own CP group";
    // One CP group per (dp_rank, tp_rank) pair. tp_rank alone is not unique:
    // with dp=2/cp=2/tp=4 the groups {0,4} and {8,12} both have tp_rank 0 and
    // would race for the same TCPStore port.
    const int32_t dp_stride = normalized_cp_size * tp_size;
    const int32_t cp_group_index =
        (global_rank / dp_stride) * tp_size + global_rank % tp_size;
    const int32_t cp_group_count = dp_size * tp_size;
    cp_group_ =
        create_process_group(global_rank,
                             cp_local_rank,
                             cp_ranks,
                             world_size,
                             normalized_cp_size,
                             port + cp_group_index + 1,
                             get_rank_table_server_host(cp_ranks.front(), host),
                             "cp_group",
                             device);
    parallel_args_->cp_group_ = cp_group_.get();
    port += cp_group_count;
  } else
#endif
  {
    // The current MLU model-side CP path spans the full DP-local rank set,
    // which is also represented by tp_group_ today. Keep a distinct CP handle
    // so an orthogonal CP x TP topology can provide its own process group.
    parallel_args_->cp_group_ = tp_group_.get();
  }

  if (dp_size > 1) {
    // A DP group varies dp_rank while preserving the full local model-shard
    // index. Under orthogonal CP that index spans cp_rank AND tp_rank, so the
    // grouping key is global_rank % (world_size / dp_size) -- matching the
    // `trans` grouping below. Keying on tp_size alone was equivalent only
    // while tp_size == world/dp; after narrowing it collides.
    const int32_t dp_group_count = world_size / dp_size;
    port_offset = global_rank % dp_group_count + 1;
    dp_local_process_group_ = create_process_group(global_rank,
                                                   world_size,
                                                   dp_size,
                                                   port + port_offset,
                                                   true,
                                                   host,
                                                   "dp_group",
                                                   device);
    parallel_args_->dp_local_process_group_ = dp_local_process_group_.get();
    port += dp_group_count;
  }

  int32_t moe_tp_size = world_size / ep_size;
  CHECK_EQ(moe_tp_size * ep_size, world_size);
  if (ep_size == 1) {
    parallel_args_->moe_tp_group_ = process_group_.get();
    parallel_args_->eplb_group_ = process_group_.get();
  } else {
    port_offset = global_rank / moe_tp_size + 1;
    std::string moe_tp_host = host;
#if defined(USE_NPU)
    if (::xllm::KernelConfig::get_instance().npu_kernel_backend() == "TORCH") {
      const int32_t moe_tp_group_start =
          (global_rank / moe_tp_size) * moe_tp_size;
      moe_tp_host = get_rank_table_server_host(moe_tp_group_start, host);
    }
#endif
    moe_tp_group_ = create_process_group(global_rank,
                                         world_size,
                                         moe_tp_size,
                                         port + port_offset,
                                         false,
                                         moe_tp_host,
                                         "moe_tp_group",
                                         device);
    parallel_args_->moe_tp_group_ = moe_tp_group_.get();
    port += ep_size;
    port_offset = global_rank % moe_tp_size + 1;
    moe_ep_group_ = create_process_group(global_rank,
                                         world_size,
                                         ep_size,
                                         port + port_offset,
                                         true,
                                         host,
                                         "moe_ep_group",
                                         device);
    parallel_args_->moe_ep_group_ = moe_ep_group_.get();
    port += moe_tp_size;
#if defined(USE_NPU)
    if (::xllm::KernelConfig::get_instance().npu_kernel_backend() == "TORCH" &&
        ::xllm::KernelConfig::get_instance().enable_fused_mc2() > 0) {
      mc2_group_ = create_process_group(global_rank,
                                        world_size,
                                        ep_size,
                                        port + port_offset,
                                        true,
                                        host,
                                        "mc2_group",
                                        device);
      parallel_args_->mc2_group_ = mc2_group_.get();
      const std::string mc2_comm_name =
          mc2_group_->hccl_comm_name(/*init_comm=*/true);
      CHECK(!mc2_comm_name.empty())
          << "Fused MC2 process group failed to initialize its HCCL "
             "communicator.";
      port += moe_tp_size;
    }
#endif
    if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
      eplb_group_ = create_process_group(global_rank,
                                         world_size,
                                         ep_size,
                                         port + port_offset,
                                         true,
                                         host,
                                         "eplb_group",
                                         device);
      parallel_args_->eplb_group_ = eplb_group_.get();
#if defined(USE_NPU)
      // Match vLLM Ascend's dynamic EPLB initialization: establish every P2P
      // communicator before EP2 registers its dispatch/combine comm domain.
      // Lazy HCCL initialization during a weight transfer can otherwise race
      // with the next MoE dispatch and temporarily invalidate group lookup.
      eplb_group_->warmup_p2p();
#endif
      port += moe_tp_size;
    }
  }

#if defined(USE_NPU)
  if (::xllm::KernelConfig::get_instance().npu_kernel_backend() == "TORCH" &&
      ::xllm::EPLBConfig::get_instance().expert_parallel_degree() == 2 &&
      ep_size == world_size) {
    auto dispatch_and_combine_comm = create_dispatch_and_combine_comm(
        global_rank, world_size, dp_size, ep_size, cp_size);
    parallel_args_->mapping_data(dispatch_and_combine_comm.mapping_data);
    parallel_args_->mapping(dispatch_and_combine_comm.mapping);
    parallel_args_->dispatchAndCombinecommDomain(
        dispatch_and_combine_comm.domain);
    parallel_args_->dispatchAndCombineHcclComm(dispatch_and_combine_comm.comm);
  }
#endif
}

const ParallelArgs* CollectiveCommunicator::parallel_args() {
  // TODO: init communicator
  return parallel_args_.get();
}

}  // namespace xllm
