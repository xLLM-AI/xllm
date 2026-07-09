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

#include "dit_collective_communicator.h"

#include "mapping_npu.h"

#if defined(USE_NPU)
#include "npu_process_group.h"
#elif defined(USE_MLU)
#include "mlu_process_group.h"
#elif defined(USE_CUDA) || defined(USE_DCU)
#include "cuda_process_group.h"
#elif defined(USE_ILU)
#include "ilu_process_group.h"
#elif defined(USE_MUSA)
#include "musa_process_group.h"
#endif
#include "common/global_flags.h"
#include "parallel_args.h"
#include "platform/device.h"
#include "process_group.h"
#include "util/net.h"
namespace xllm {

DiTCollectiveCommunicator::DiTCollectiveCommunicator(int32_t global_rank,
                                                     int32_t world_size,
                                                     int32_t dit_dp_size,
                                                     int32_t dit_tp_size,
                                                     int32_t dit_sp_size,
                                                     int32_t dit_cfg_size,
                                                     int32_t dit_vae_size)
    : CollectiveCommunicatorBase(global_rank, world_size) {
  parallel_args_ = std::make_unique<ParallelArgs>(global_rank,
                                                  world_size,
                                                  dit_dp_size,
                                                  dit_tp_size,
                                                  dit_sp_size,
                                                  dit_cfg_size,
                                                  dit_vae_size,
                                                  /*process_group=*/nullptr);
  DiTMapping::Options dit_mapping_options;
  dit_mapping_options.dit_tp_size(dit_tp_size)
      .dit_sp_size(dit_sp_size)
      .dit_cfg_size(dit_cfg_size)
      .dit_dp_size(dit_dp_size)
      .dit_vae_size(dit_vae_size);
  dit_mapping_ = std::make_unique<DiTMapping>(
      world_size, global_rank, dit_mapping_options);
}

void DiTCollectiveCommunicator::create_process_groups(
    const std::string& master_addr,
    const torch::Device& device) {
  Device device_(device);
  device_.set_device();
  net::parse_host_port_from_addr(master_addr, host_, port_);

  process_group_ = create_process_group(global_rank_,
                                        world_size_,
                                        world_size_,
                                        ++port_,
                                        false,
                                        host_,
                                        "world_group",
                                        device);

  parallel_args_->process_group_ = process_group_.get();
  parallel_args_->dit_tp_group_ =
      create_process_group_by_type("tp", dit_tp_group_, device);
  parallel_args_->dit_sp_group_ =
      create_process_group_by_type("sp", dit_sp_group_, device);
  parallel_args_->dit_cfg_group_ =
      create_process_group_by_type("cfg", dit_cfg_group_, device);
  parallel_args_->dit_dp_group_ =
      create_process_group_by_type("dp", dit_dp_group_, device);
  parallel_args_->dit_vae_group_ =
      create_process_group_by_type("vae", dit_vae_group_, device);
}

ProcessGroup* DiTCollectiveCommunicator::create_process_group_by_type(
    const std::string& group_type,
    std::unique_ptr<ProcessGroup>& member_group,
    const torch::Device& device) {
  int32_t group_size = parallel_args_->get_group_size_by_type(group_type);
  if (group_size > 1 && dit_mapping_) {
    auto parallel_info = dit_mapping_->get_parallel_info(group_type);
    auto group_id = parallel_info.current_group_id();
    auto num_group = parallel_info.num_group();
    auto local_rank = parallel_info.rank();
    auto& rank_per_group = parallel_info.rank_per_group()[group_id];
    int port_offset = group_id + 1;
#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
    member_group = std::move(create_process_group(global_rank_,
                                                  local_rank,
                                                  rank_per_group,
                                                  world_size_,
                                                  group_size,
                                                  port_ + port_offset,
                                                  host_,
                                                  group_type + "_group",
                                                  device));
#else
    LOG(FATAL)
        << "create_process_group function is used by DiT models, since "
           "the DiT communication group "
        << "info have already been calculated by rank_generator, we only "
           "need to pass the "
        << "info to create the process groups. For any device that want "
           "to reuse the "
        << "function and dit process groups, please implement the "
           "corresponding "
        << "ProcessGroupImpl construct function. ";
#endif
    port_ += num_group;
  }
  return member_group.get();
}

const ParallelArgs* DiTCollectiveCommunicator::parallel_args() {
  // TODO: init communicator
  return parallel_args_.get();
}

}  // namespace xllm
