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
                                                     int32_t dit_cfg_size)
    : CollectiveCommunicatorBase(global_rank, world_size) {
  parallel_args_ = std::make_unique<ParallelArgs>(global_rank,
                                                  world_size,
                                                  dit_dp_size,
                                                  dit_tp_size,
                                                  dit_sp_size,
                                                  dit_cfg_size,
                                                  /*process_group=*/nullptr);
  DiTMapping::Options dit_mapping_options;
  dit_mapping_options.dit_tp_size(dit_tp_size)
      .dit_sp_size(dit_sp_size)
      .dit_cfg_size(dit_cfg_size)
      .dit_dp_size(dit_dp_size);
  dit_mapping_ = std::make_unique<DiTMapping>(
      world_size, global_rank, dit_mapping_options);
}

void DiTCollectiveCommunicator::create_process_groups(
    const std::string& master_addr,
    const torch::Device& device) {
  Device device_(device);
  device_.set_device();
  std::string host;
  int32_t port;
  net::parse_host_port_from_addr(master_addr, host, port);

  int32_t global_rank = parallel_args_->rank();
  int32_t world_size = parallel_args_->world_size();
  int32_t dp_size = parallel_args_->dp_size();
  int32_t tp_size = parallel_args_->tp_size();
  int32_t sp_size = parallel_args_->sp_size();
  int32_t cfg_size = parallel_args_->cfg_size();

  process_group_ = create_process_group(global_rank,
                                        world_size,
                                        world_size,
                                        ++port,
                                        false,
                                        host,
                                        "world_group",
                                        device);

  parallel_args_->process_group_ = process_group_.get();

  if (tp_size > 1 && dit_mapping_) {
    auto tp_parallel_info = dit_mapping_->get_parallel_info("tp");
    auto group_id = tp_parallel_info.current_group_id();
    auto num_group = tp_parallel_info.num_group();
    auto local_rank = tp_parallel_info.rank();
    auto& rank_per_group = tp_parallel_info.rank_per_group()[group_id];
    int port_offset = group_id + 1;
#if defined(USE_NPU) || defined(USE_MLU)
    dit_tp_group_ = create_process_group(global_rank,
                                         local_rank,
                                         rank_per_group,
                                         world_size,
                                         tp_size,
                                         port + port_offset,
                                         host,
                                         "tp_group",
                                         device);
#endif
    parallel_args_->dit_tp_group_ = dit_tp_group_.get();
    port += num_group;
  }

  if (sp_size > 1 && dit_mapping_) {
    auto sp_parallel_info = dit_mapping_->get_parallel_info("sp");
    auto group_id = sp_parallel_info.current_group_id();
    auto num_group = sp_parallel_info.num_group();
    auto local_rank = sp_parallel_info.rank();
    auto& rank_per_group = sp_parallel_info.rank_per_group()[group_id];
    int port_offset = group_id + 1;
#if defined(USE_NPU) || defined(USE_MLU)
    dit_sp_group_ = create_process_group(global_rank,
                                         local_rank,
                                         rank_per_group,
                                         world_size,
                                         sp_size,
                                         port + port_offset,
                                         host,
                                         "sp_group",
                                         device);
#endif
    parallel_args_->dit_sp_group_ = dit_sp_group_.get();
    port += num_group;
  }

  if (cfg_size > 1 && dit_mapping_) {
    auto cfg_parallel_info = dit_mapping_->get_parallel_info("cfg");
    auto group_id = cfg_parallel_info.current_group_id();
    auto num_group = cfg_parallel_info.num_group();
    auto local_rank = cfg_parallel_info.rank();
    auto& rank_per_group = cfg_parallel_info.rank_per_group()[group_id];
    int port_offset = group_id + 1;
#if defined(USE_NPU) || defined(USE_MLU)
    dit_cfg_group_ = create_process_group(global_rank,
                                          local_rank,
                                          rank_per_group,
                                          world_size,
                                          cfg_size,
                                          port + port_offset,
                                          host,
                                          "cfg_group",
                                          device);
#endif
    parallel_args_->dit_cfg_group_ = dit_cfg_group_.get();
    port += num_group;
  }

  if (dp_size > 1 && dit_mapping_) {
    auto dp_parallel_info = dit_mapping_->get_parallel_info("dp");
    auto group_id = dp_parallel_info.current_group_id();
    auto num_group = dp_parallel_info.num_group();
    auto local_rank = dp_parallel_info.rank();
    auto& rank_per_group = dp_parallel_info.rank_per_group()[group_id];
    int port_offset = group_id + 1;
#if defined(USE_NPU) || defined(USE_MLU)
    dit_dp_group_ = create_process_group(global_rank,
                                         local_rank,
                                         rank_per_group,
                                         world_size,
                                         dp_size,
                                         port + port_offset,
                                         host,
                                         "dp_group",
                                         device);
#endif
    parallel_args_->dit_dp_group_ = dit_dp_group_.get();
    port += num_group;
  }
}

const ParallelArgs* DiTCollectiveCommunicator::parallel_args() {
  // TODO: init communicator
  return parallel_args_.get();
}

}  // namespace xllm
