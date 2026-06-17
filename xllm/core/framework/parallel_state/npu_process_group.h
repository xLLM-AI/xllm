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

#include <hccl/hccl_types.h>
#include <torch_npu/csrc/core/npu/NPUEvent.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "core/common/global_flags.h"
#include "hccl/hccl.h"
#include "process_group.h"

namespace xllm {

class ProcessGroupImpl : public ProcessGroup {
 public:
  // Constructor.
  ProcessGroupImpl(int rank,
                   int world_size,
                   const torch::Device& device,
                   HcclComm comm);

  ProcessGroupImpl(int rank,
                   int world_size,
                   int rank_size,
                   int port,
                   bool trans,
                   const std::string& host,
                   const std::string& group_name,
                   const torch::Device& device);

  ProcessGroupImpl(int32_t global_rank,
                   int32_t local_rank,
                   const std::vector<int32_t>& group_ranks,
                   int32_t world_size,
                   int32_t rank_size,
                   int32_t port,
                   const std::string& host,
                   const std::string& group_name,
                   const torch::Device& device);

  // Destructor.
  ~ProcessGroupImpl() override;

  std::string hccl_comm_name(bool init_comm = true) override;

 private:
  HcclComm comm_ = nullptr;
  c10_npu::NPUStream comm_stream_;
};

// TODO: LOG HcclGetErrorString(r)
#if defined(USE_NPU)
#define HCCLCHECK(cmd)                     \
  do {                                     \
    HcclResult r = cmd;                    \
    if (r != HCCL_SUCCESS) {               \
      LOG(FATAL) << "Failed, HCCL error."; \
    }                                      \
  } while (0)
#endif
}  // namespace xllm
