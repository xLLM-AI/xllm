/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

namespace xllm {

class CollectiveCommunicator : public CollectiveCommunicatorBase {
 public:
  CollectiveCommunicator(int global_rank,
                         int world_size,
                         int dp_size,
                         int ep_size,
                         int cp_size);
  ~CollectiveCommunicator() = default;

  void create_process_groups(const std::string& master_addr,
                             const torch::Device& device) override;

  // init communicator and return parallel args.
  const ParallelArgs* parallel_args() override;

 private:
  std::unique_ptr<ParallelArgs> parallel_args_;
  std::unique_ptr<ProcessGroup> process_group_;
  std::unique_ptr<ProcessGroup> dp_local_process_group_;
  std::unique_ptr<ProcessGroup> tp_group_;
  std::unique_ptr<ProcessGroup> encoder_dp_group_;
  std::unique_ptr<ProcessGroup> single_rank_group_;
  // Reserved owner for a future standalone SP group. The current code path
  // aliases ParallelArgs::sp_group_ to tp_group_ instead of constructing a
  // separate communicator, so this stays empty for now.
  std::unique_ptr<ProcessGroup> sp_group_;
  std::unique_ptr<ProcessGroup> moe_tp_group_;
  std::unique_ptr<ProcessGroup> moe_ep_group_;
};

}  // namespace xllm
