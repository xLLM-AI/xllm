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

#pragma once

#include "collective_communicator_base.h"
#include "dit_mapping.h"

namespace xllm {

class DiTCollectiveCommunicator : public CollectiveCommunicatorBase {
 public:
  DiTCollectiveCommunicator(int32_t global_rank,
                            int32_t world_size,
                            int32_t dit_dp_size,
                            int32_t dit_tp_size,
                            int32_t dit_sp_size,
                            int32_t dit_cfg_size,
                            int32_t dit_vae_size);

  ~DiTCollectiveCommunicator() = default;

  void create_process_groups(const std::string& master_addr,
                             const torch::Device& device) override;

  // init communicator and return parallel args.
  const ParallelArgs* parallel_args() override;

  ProcessGroup* create_process_group_by_type(
      const std::string& group_type,
      std::unique_ptr<ProcessGroup>& member_group,
      const torch::Device& device);

 private:
  int32_t port_ = 0;
  std::string host_ = "";
  std::unique_ptr<DiTMapping> dit_mapping_{nullptr};
  std::unique_ptr<ParallelArgs> parallel_args_;
  std::unique_ptr<ProcessGroup> process_group_;
  std::unique_ptr<ProcessGroup> dit_tp_group_;
  std::unique_ptr<ProcessGroup> dit_sp_group_;
  std::unique_ptr<ProcessGroup> dit_dp_group_;
  std::unique_ptr<ProcessGroup> dit_cfg_group_;
  std::unique_ptr<ProcessGroup> dit_vae_group_;
};

}  // namespace xllm
