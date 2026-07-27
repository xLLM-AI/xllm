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

#include <torch/torch.h>

#include <memory>
#include <string>

#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/parallel_state/process_group.h"

namespace xllm {

class CollectiveCommunicatorBase {
 public:
  CollectiveCommunicatorBase(int global_rank, int world_size)
      : global_rank_(global_rank), world_size_(world_size) {}

  virtual ~CollectiveCommunicatorBase() = default;

  virtual void create_process_groups(const std::string& master_addr,
                                     const torch::Device& device) = 0;

  virtual const ParallelArgs* parallel_args() = 0;

  int get_global_rank() const { return global_rank_; }
  int get_world_size() const { return world_size_; }

 protected:
  int global_rank_;
  int world_size_;
};

}  // namespace xllm
