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

#include "collective_communicator.h"

#include "mapping_npu.h"

#if defined(USE_NPU)
#include "npu_process_group.h"
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
  // MappingNPU falls back to cp_size (legacy byte-equivalent). >0 -> propagate
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

  // Compute section bases for the master-allocated port list. Every rank
  // computes the same offsets given (world_size, dp_size, ep_size,
  // enable_mm_encoder_dp), so the (section, group_index) -> slot mapping is
  // identical on master and workers. group_ports_[slot] is then used as the
  // TCPStore port for the corresponding process group, which sidesteps the
  // EADDRINUSE race that the old "master_port + offset" scheme hits inside
  // the OS ephemeral range.
  //
  // When group_ports_ is empty (legacy callers / DiT) we fall back to the
  // master_port + offset formula via use_master_ports below, preserving
  // byte-for-byte behaviour for those paths.
  const bool use_master_ports = !group_ports_.empty();
  const bool encoder_dp_enabled =
      ::xllm::ParallelConfig::get_instance().enable_mm_encoder_dp();
  const int32_t tp_size_pre = world_size / dp_size;
  const int32_t moe_tp_size_pre = world_size / ep_size;

  const int32_t kEncoderDpBase = 0;
  const int32_t kWorldBase = encoder_dp_enabled ? dp_size : 0;
  const int32_t kTpBase = kWorldBase + 1;
  const int32_t kSingleRankBase = kTpBase + dp_size;
  const int32_t kDpLocalBase =
      kSingleRankBase + (tp_size_pre > 1 ? world_size : 0);
  const int32_t kMoeTpBase = kDpLocalBase + (dp_size > 1 ? tp_size_pre : 0);
  const int32_t kMoeEpBase = kMoeTpBase + (ep_size > 1 ? ep_size : 0);

  auto resolve_port =
      [&](int32_t section_base, int32_t group_index, int32_t legacy_port) {
        if (!use_master_ports) {
          return legacy_port;
        }
        const int32_t slot = section_base + group_index;
        CHECK_LT(slot, static_cast<int32_t>(group_ports_.size()))
            << "Master-allocated TCPStore port slot " << slot
            << " out of range (size " << group_ports_.size() << ").";
        return group_ports_[slot];
      };

  int32_t port_offset = 0;

  // Encoder DP is used by multi-modal models to parallelize vision encoder
  // work inside each language-model TP group. The rank set matches the TP
  // group, but each rank runs a full encoder on different multi-modal items.
  if (encoder_dp_enabled) {
    const int32_t encoder_dp_size = world_size / dp_size;
    const int32_t encoder_dp_group_index = global_rank / encoder_dp_size;
    port_offset = encoder_dp_group_index + 1;
    encoder_dp_group_ = create_process_group(
        global_rank,
        world_size,
        encoder_dp_size,
        resolve_port(
            kEncoderDpBase, encoder_dp_group_index, port + port_offset),
        /*trans=*/false,
        host,
        "encoder_dp_group",
        device);
    parallel_args_->encoder_dp_group_ = encoder_dp_group_.get();
    port += dp_size;
  }

#if defined(USE_NPU)
  if (::xllm::KernelConfig::get_instance().npu_kernel_backend() == "ATB") {
    return;
  }
#endif

  process_group_ = create_process_group(global_rank,
                                        world_size,
                                        world_size,
                                        resolve_port(kWorldBase, 0, ++port),
                                        /*trans=*/false,
                                        host,
                                        "world_group",
                                        device);
  parallel_args_->process_group_ = process_group_.get();

  int32_t tp_size = world_size / dp_size;
  CHECK_EQ(tp_size * dp_size, world_size);
  const int32_t tp_group_index = global_rank / tp_size;
  port_offset = tp_group_index + 1;
  std::string tp_host = host;
#if defined(USE_NPU)
  if (::xllm::KernelConfig::get_instance().npu_kernel_backend() == "TORCH" &&
      dp_size > 1) {
    const int32_t tp_group_start = (global_rank / tp_size) * tp_size;
    tp_host = get_rank_table_server_host(tp_group_start, host);
  }
#endif
  tp_group_ = create_process_group(
      global_rank,
      world_size,
      tp_size,
      resolve_port(kTpBase, tp_group_index, port + port_offset),
      /*trans=*/false,
      tp_host,
      "tp_group",
      device);
  parallel_args_->tp_group_ = tp_group_.get();
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
        resolve_port(
            kSingleRankBase,
            global_rank,
            port + dp_size + single_rank_group_port_gap + global_rank + 1),
        /*trans=*/false,
        host,
        "single_rank_group",
        device);
    parallel_args_->single_rank_group_ = single_rank_group_.get();
    single_rank_group_count = world_size;
  } else {
    parallel_args_->single_rank_group_ = tp_group_.get();
  }
  // The current MLU model-side CP path spans the full DP-local rank set, which
  // is also represented by tp_group_ today. Keep a distinct CP handle so a
  // future orthogonal CP x TP topology can provide its own process group.
  parallel_args_->cp_group_ = tp_group_.get();
  port += dp_size + single_rank_group_port_gap + single_rank_group_count;

  if (dp_size > 1) {
    const int32_t dp_local_group_index = global_rank % tp_size;
    port_offset = dp_local_group_index + 1;
    dp_local_process_group_ = create_process_group(
        global_rank,
        world_size,
        dp_size,
        resolve_port(kDpLocalBase, dp_local_group_index, port + port_offset),
        /*trans=*/true,
        host,
        "dp_group",
        device);
    parallel_args_->dp_local_process_group_ = dp_local_process_group_.get();
    port += tp_size;
  }

  int32_t moe_tp_size = world_size / ep_size;
  CHECK_EQ(moe_tp_size * ep_size, world_size);
  if (ep_size == 1) {
    parallel_args_->moe_tp_group_ = process_group_.get();
  } else {
    const int32_t moe_tp_group_index = global_rank / moe_tp_size;
    port_offset = moe_tp_group_index + 1;
    std::string moe_tp_host = host;
#if defined(USE_NPU)
    if (::xllm::KernelConfig::get_instance().npu_kernel_backend() == "TORCH") {
      const int32_t moe_tp_group_start =
          (global_rank / moe_tp_size) * moe_tp_size;
      moe_tp_host = get_rank_table_server_host(moe_tp_group_start, host);
    }
#endif
    moe_tp_group_ = create_process_group(
        global_rank,
        world_size,
        moe_tp_size,
        resolve_port(kMoeTpBase, moe_tp_group_index, port + port_offset),
        /*trans=*/false,
        moe_tp_host,
        "moe_tp_group",
        device);
    parallel_args_->moe_tp_group_ = moe_tp_group_.get();
    port += ep_size;
    const int32_t moe_ep_group_index = global_rank % moe_tp_size;
    port_offset = moe_ep_group_index + 1;
    moe_ep_group_ = create_process_group(
        global_rank,
        world_size,
        ep_size,
        resolve_port(kMoeEpBase, moe_ep_group_index, port + port_offset),
        /*trans=*/true,
        host,
        "moe_ep_group",
        device);
    parallel_args_->moe_ep_group_ = moe_ep_group_.get();
    port += moe_tp_size;
  }

  parallel_args_->python_tp_rendezvous_host_ = host;
  parallel_args_->python_tp_rendezvous_port_ = port + tp_group_index + 1;
  CHECK_LE(parallel_args_->python_tp_rendezvous_port_, 65535)
      << "No TCP port remains for the Python TP rendezvous.";

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
